// Document parsers: turn common file formats into model-friendly text.
// Strategy: pure-Go libraries only (CGO would kill the "go build and go"
// property of the template); scans and complex layouts are left to the
// vision model via image attachments.
package tools

import (
	"archive/zip"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/ledongthuc/pdf"
	"github.com/xuri/excelize/v2"
)

// maxSheetRows caps one spreadsheet sheet; larger sheets are cut with a
// note so the model knows to narrow its read.
const maxSheetRows = 200

// Limits on what a zip-based document may inflate to: the whole archive and
// its largest XML part (a zip bomb is a few KB that unpack to gigabytes).
const (
	maxUnzippedBytes    = 128 << 20
	maxUnzippedXMLBytes = 32 << 20
)

// ParseDocument converts raw bytes of a known extension into text.
// Unknown extensions return the bytes as-is when they look textual,
// base64 otherwise. The note explains format-specific caveats.
func ParseDocument(ext string, raw []byte) (content string, note string, err error) {
	// The parsers run on bytes from outside (a downloaded file, an upload);
	// the PDF library in particular panics on malformed input.
	defer func() {
		if r := recover(); r != nil {
			content, note, err = "", "", fmt.Errorf("%s: malformed document (%v)", strings.TrimPrefix(ext, "."), r)
		}
	}()
	switch ext {
	case ".xlsx":
		text, err := parseXLSX(raw)
		return text, "first " + fmt.Sprint(maxSheetRows) + " rows per sheet", err
	case ".pdf":
		text, err := parsePDF(raw)
		return text, "text layer only; for scanned pages attach page images for vision", err
	case ".docx":
		text, err := parseDOCX(raw)
		return text, "", err
	case ".csv":
		return normalizeCSV(raw)
	case ".json":
		return compactJSON(raw)
	default:
		if isLikelyText(raw) {
			return string(raw), "", nil
		}
		return base64OrEmpty(raw), "binary content, base64-encoded", nil
	}
}

// parseXLSX renders every sheet as pipe-separated rows.
func parseXLSX(raw []byte) (string, error) {
	f, err := excelize.OpenReader(bytes.NewReader(raw), excelize.Options{
		UnzipSizeLimit:    maxUnzippedBytes,
		UnzipXMLSizeLimit: maxUnzippedXMLBytes,
	})
	if err != nil {
		return "", fmt.Errorf("xlsx: %w", err)
	}
	defer f.Close()

	var sb strings.Builder
	for _, sheet := range f.GetSheetList() {
		rows, err := f.GetRows(sheet)
		if err != nil {
			return "", fmt.Errorf("xlsx: sheet %q: %w", sheet, err)
		}
		fmt.Fprintf(&sb, "## sheet: %s\n", sheet)
		for i, row := range rows {
			if i >= maxSheetRows {
				fmt.Fprintf(&sb, "... (%d more rows)\n", len(rows)-maxSheetRows)
				break
			}
			sb.WriteString(strings.Join(row, " | "))
			sb.WriteByte('\n')
		}
		sb.WriteByte('\n')
	}
	return strings.TrimSpace(sb.String()), nil
}

// parsePDF extracts the text layer (pure Go).
func parsePDF(raw []byte) (string, error) {
	reader := bytes.NewReader(raw)
	doc, err := pdf.NewReader(reader, int64(len(raw)))
	if err != nil {
		return "", fmt.Errorf("pdf: %w", err)
	}

	var sb strings.Builder
	n := doc.NumPage()
	for i := 1; i <= n; i++ {
		page := doc.Page(i)
		if page.V.IsNull() {
			continue
		}
		text, err := page.GetPlainText(nil)
		if err != nil {
			continue // unreadable page — skip, don't fail the file
		}
		fmt.Fprintf(&sb, "## page %d/%d\n%s\n", i, n, strings.TrimSpace(text))
	}
	if sb.Len() == 0 {
		return "", fmt.Errorf("pdf: no text layer (likely a scan)")
	}
	return sb.String(), nil
}

var (
	reDocxText = regexp.MustCompile(`(?s)<w:t[^>]*>(.*?)</w:t>`)
	reDocxBr   = regexp.MustCompile(`</w:p>`)
)

// parseDOCX reads word/document.xml out of the zip container.
func parseDOCX(raw []byte) (string, error) {
	zr, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return "", fmt.Errorf("docx: %w", err)
	}

	var docXML []byte
	for _, f := range zr.File {
		if f.Name == "word/document.xml" {
			rc, err := f.Open()
			if err != nil {
				return "", fmt.Errorf("docx: open document.xml: %w", err)
			}
			defer rc.Close()
			// The archive is attacker-sized: a few KB can inflate to
			// gigabytes, so read one byte past the limit and refuse.
			docXML, err = io.ReadAll(io.LimitReader(rc, maxUnzippedXMLBytes+1))
			if err != nil {
				return "", fmt.Errorf("docx: read document.xml: %w", err)
			}
			if len(docXML) > maxUnzippedXMLBytes {
				return "", fmt.Errorf("docx: document.xml is larger than %d MB unpacked", maxUnzippedXMLBytes>>20)
			}
			break
		}
	}
	if docXML == nil {
		return "", fmt.Errorf("docx: word/document.xml not found")
	}

	// Paragraph breaks become newlines, w:t runs become text.
	text := reDocxBr.ReplaceAllString(string(docXML), "\n")
	matches := reDocxText.FindAllStringSubmatch(text, -1)
	var sb strings.Builder
	for _, m := range matches {
		sb.WriteString(unescapeXML(m[1]))
	}
	out := strings.TrimSpace(sb.String())
	if out == "" {
		return "", fmt.Errorf("docx: no text found")
	}
	return out, nil
}

func unescapeXML(s string) string {
	r := strings.NewReplacer("&lt;", "<", "&gt;", ">", "&quot;", `"`, "&apos;", "'", "&amp;", "&")
	return r.Replace(s)
}

// normalizeCSV reads a CSV as records joined by pipes (robust to quotes).
func normalizeCSV(raw []byte) (string, string, error) {
	recs, err := csv.NewReader(bytes.NewReader(raw)).ReadAll()
	if err != nil {
		return "", "", fmt.Errorf("csv: %w", err)
	}
	var sb strings.Builder
	for _, rec := range recs {
		sb.WriteString(strings.Join(rec, " | "))
		sb.WriteByte('\n')
	}
	return sb.String(), "", nil
}

// compactJSON validates and re-renders JSON with stable indentation.
func compactJSON(raw []byte) (string, string, error) {
	var buf bytes.Buffer
	if err := json.Indent(&buf, raw, "", "  "); err != nil {
		// Not valid JSON — return as text; the model can cope.
		return string(raw), "", nil
	}
	return buf.String(), "", nil
}

// isLikelyText reports whether the bytes look like UTF-8-ish text (no NUL
// bytes, mostly printable).
func isLikelyText(b []byte) bool {
	if bytes.IndexByte(b, 0) >= 0 {
		return false
	}
	nonPrintable := 0
	limit := len(b)
	if limit > 4096 {
		limit = 4096
	}
	for _, r := range b[:limit] {
		if r < 9 || (r > 13 && r < 32) {
			nonPrintable++
		}
	}
	return nonPrintable*20 < limit // <5% control characters
}
