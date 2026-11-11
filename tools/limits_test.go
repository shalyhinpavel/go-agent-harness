package tools

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func zipWith(t *testing.T, name string, size int) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	w, err := zw.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	chunk := bytes.Repeat([]byte("<w:t>a</w:t>"), 1<<12)
	for written := 0; written < size; written += len(chunk) {
		w.Write(chunk)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestParseDOCX_ZipBombRefused(t *testing.T) {
	bomb := zipWith(t, "word/document.xml", maxUnzippedXMLBytes+1<<20)
	if len(bomb) > 1<<20 {
		t.Fatalf("fixture should be tiny, is %d bytes", len(bomb))
	}
	if _, err := parseDOCX(bomb); err == nil || !strings.Contains(err.Error(), "unpacked") {
		t.Fatalf("an oversized document.xml must be refused, got %v", err)
	}
}

func TestParseDocument_MalformedInputDoesNotPanic(t *testing.T) {
	junk := []byte("%PDF-1.7\n1 0 obj\n<< /Type /Pages /Kids [ 2 0 R ] /Count 1 >>\nendobj\ntrailer << /Root 1 0 R >>\n%%EOF")
	for _, ext := range []string{".pdf", ".docx", ".xlsx"} {
		if _, _, err := ParseDocument(ext, junk); err == nil {
			t.Errorf("%s: garbage must be an error", ext)
		}
	}
}

func TestReadWorkspaceFile_OversizedFiles(t *testing.T) {
	dir := t.TempDir()
	big := make([]byte, maxReadBytes+1024)
	for i := range big {
		big[i] = 'x'
	}
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), big, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "big.pdf"), big, 0o644); err != nil {
		t.Fatal(err)
	}

	res, err := readWorkspaceFile(dir, "big.txt")
	if err != nil {
		t.Fatal(err)
	}
	data := res.Data.(map[string]any)
	if !strings.Contains(data["note"].(string), "first 8 MB") {
		t.Errorf("a cut text file must say so: %v", data["note"])
	}
	if _, err := readWorkspaceFile(dir, "big.pdf"); err == nil || !strings.Contains(err.Error(), "too big") {
		t.Errorf("an oversized container must be refused, not cut into a broken file: %v", err)
	}
}

func TestWorkspace_SymlinkCannotLeave(t *testing.T) {
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("TOPSECRET"), 0o644); err != nil {
		t.Fatal(err)
	}
	ws := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(ws, "link")); err != nil {
		t.Fatal(err)
	}

	if _, err := readWorkspaceFile(ws, "link/secret.txt"); err == nil {
		t.Error("read through a symlink out of the workspace must fail")
	}
	if _, err := writeWorkspaceFile(ws, "link/pwned.txt", "x"); err == nil {
		t.Error("write through a symlink out of the workspace must fail")
	}
	if _, err := os.Stat(filepath.Join(outside, "pwned.txt")); err == nil {
		t.Error("the file landed outside the workspace")
	}
	if err := StoreFile(ws, "link", []byte("x")); err == nil {
		t.Error("an attachment named like an outward symlink must not write through it")
	}
}

func TestWorkspace_SymlinkInsideStillWorks(t *testing.T) {
	ws := t.TempDir()
	if _, err := writeWorkspaceFile(ws, "docs/a.txt", "hello"); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("docs", filepath.Join(ws, "alias")); err != nil {
		t.Fatal(err)
	}
	res, err := readWorkspaceFile(ws, "alias/a.txt")
	if err != nil || res.Data.(map[string]any)["content"] != "hello" {
		t.Fatalf("a symlink that stays inside is fine: %v %v", res, err)
	}
}

func TestListWorkspace_IncludesSubdirectories(t *testing.T) {
	ws := t.TempDir()
	for _, p := range []string{"a.txt", "sub/b.txt", "sub/deep/c.txt"} {
		if _, err := writeWorkspaceFile(ws, p, "x"); err != nil {
			t.Fatal(err)
		}
	}
	res, err := listWorkspace(ws)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, f := range res.Data.(map[string]any)["files"].([]map[string]any) {
		if !f["dir"].(bool) {
			names = append(names, f["name"].(string))
		}
	}
	want := "a.txt sub/b.txt sub/deep/c.txt"
	if strings.Join(names, " ") != want {
		t.Fatalf("files = %v, want %s", names, want)
	}
}
