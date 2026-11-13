package tools

import (
	"context"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// WebFetchName is the web fetch tool's name.
const WebFetchName = "web_fetch"

// webFetchUserAgent identifies the harness honestly. Many sites refuse
// anonymous clients; set your own agent name and contact URL in
// WebFetchConfig.UserAgent for anything beyond local testing.
const webFetchUserAgent = "go-agent-harness/0.1 (+https://github.com/shalyhinpavel/go-agent-harness)"

// WebFetchConfig parameterizes the web_fetch tool. Zero values take the
// defaults noted on each field.
type WebFetchConfig struct {
	TimeoutS     int    `yaml:"timeout_s"`     // per-request deadline; 0 = 15 s
	MaxBytes     int    `yaml:"max_bytes"`     // response body cap; 0 = 2 MB
	PageChars    int    `yaml:"page_chars"`    // characters of page text per call; 0 = 12000
	BlockPrivate bool   `yaml:"block_private"` // refuse loopback/private/link-local targets (anti-SSRF)
	UserAgent    string `yaml:"user_agent"`    // "" = the harness default; set your own for real use
}

// NewWebFetch builds the web_fetch tool: GET a URL and hand the model the
// page as readable text, with its links kept as [text](absolute URL) so it
// can follow them. There is no search tool in the core: the model starts
// from addresses it knows (sites, their own search pages, docs) and
// navigates from there. Plug a real search API in as your own tool or an
// MCP server.
//
// The HTML extraction is deliberately simple — a starting point, not a
// readability engine. Long pages are served in windows of PageChars; the
// model asks for the next one with offset.
func NewWebFetch(cfg WebFetchConfig) *agent.Tool {
	timeout := time.Duration(cfg.TimeoutS) * time.Second
	if timeout <= 0 {
		timeout = 15 * time.Second
	}
	maxBytes := cfg.MaxBytes
	if maxBytes <= 0 {
		maxBytes = 2 * 1024 * 1024
	}
	pageChars := cfg.PageChars
	if pageChars <= 0 {
		pageChars = 12000
	}
	userAgent := cfg.UserAgent
	if userAgent == "" {
		userAgent = webFetchUserAgent
	}

	// Start from the default transport (idle and TLS timeouts, HTTP/2) and
	// replace only the dialer.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if cfg.BlockPrivate {
		// Through a proxy the dial goes to the proxy, not the target, and
		// the address guard below would check the wrong host.
		transport.Proxy = nil
	}
	// The Control hook fires after DNS resolution on every dial (redirects
	// included), so private targets are refused even when the hostname
	// looked public.
	transport.DialContext = (&net.Dialer{
		Timeout:   timeout,
		KeepAlive: 30 * time.Second,
		Control: func(network, address string, _ syscall.RawConn) error {
			if !cfg.BlockPrivate {
				return nil
			}
			return guardPrivateAddress(address)
		},
	}).DialContext
	client := &http.Client{Timeout: timeout, Transport: transport}

	return &agent.Tool{
		Name:     WebFetchName,
		ReadOnly: true,
		Description: "Fetch a web page or file by URL and return its text. HTML comes back as readable text with links as [text](url) — follow them with another call. " +
			"There is no search engine: start from a URL you know (a site, its own search page such as https://en.wikipedia.org/w/index.php?search=..., documentation) and navigate. " +
			"Long pages come in parts: call again with next_offset to continue. PDF, DOCX, XLSX, CSV and JSON are converted to text.",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"url": map[string]any{
					"type":        "string",
					"description": "Absolute http:// or https:// URL",
				},
				"offset": map[string]any{
					"type":        "integer",
					"description": "Character offset into the page text, from a previous call's next_offset (default 0)",
				},
			},
			"required": []string{"url"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			rawURL, _ := args["url"].(string)
			rawURL = strings.TrimSpace(rawURL)
			if !strings.HasPrefix(rawURL, "http://") && !strings.HasPrefix(rawURL, "https://") {
				return nil, fmt.Errorf("web_fetch: url must start with http:// or https://")
			}
			offset := 0
			if v, ok := args["offset"].(float64); ok && v > 0 {
				offset = int(v)
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
			if err != nil {
				return nil, fmt.Errorf("web_fetch: bad url: %w", err)
			}
			req.Header.Set("User-Agent", userAgent)
			req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8")

			resp, err := client.Do(req)
			if err != nil {
				return nil, fmt.Errorf("web_fetch: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode >= 400 {
				return nil, statusError(resp.StatusCode)
			}

			body, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)+1))
			if err != nil {
				return nil, fmt.Errorf("web_fetch: read body: %w", err)
			}
			cut := len(body) > maxBytes
			if cut {
				body = body[:maxBytes]
			}

			final := resp.Request.URL // after redirects
			page, err := pageText(resp.Header.Get("Content-Type"), final, body)
			if err != nil {
				return nil, err
			}

			data := map[string]any{"url": final.String(), "status": resp.StatusCode}
			if page.title != "" {
				data["title"] = page.title
			}
			if page.note != "" {
				data["note"] = page.note
			}
			if cut {
				data["body_cut"] = fmt.Sprintf("the response was cut at %d bytes", maxBytes)
			}
			content, next, total := window(page.text, offset, pageChars)
			data["content"] = content
			data["total_chars"] = total
			if next > 0 {
				data["next_offset"] = next
			}
			return &agent.ToolResult{Data: data}, nil
		},
	}
}

// statusError turns an HTTP failure into advice the model can act on.
func statusError(code int) error {
	switch code {
	case http.StatusNotFound, http.StatusGone:
		return fmt.Errorf("web_fetch: %d not found — the address may be outdated or guessed; find the page through a link on the site instead", code)
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusTooManyRequests, http.StatusUnavailableForLegalReasons:
		return fmt.Errorf("web_fetch: %d — the site refuses automated access or rate-limits it; use another source", code)
	default:
		return fmt.Errorf("web_fetch: HTTP %d", code)
	}
}

type fetchedPage struct {
	title string
	text  string
	note  string
}

// pageText converts a response body to model-readable text by its media
// type: HTML is extracted, documents go through the file parsers, other
// text passes through, binary content is refused.
func pageText(contentType string, base *url.URL, body []byte) (fetchedPage, error) {
	media, _, _ := mime.ParseMediaType(contentType)
	if media == "" {
		media = http.DetectContentType(body)
		media, _, _ = mime.ParseMediaType(media)
	}

	docExt := map[string]string{
		"application/pdf": ".pdf",
		"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":       ".xlsx",
		"text/csv":         ".csv",
		"application/json": ".json",
	}
	switch {
	case media == "text/html" || media == "application/xhtml+xml":
		title, text := htmlToText(string(body), base)
		return fetchedPage{title: title, text: text}, nil
	case docExt[media] != "":
		text, note, err := ParseDocument(docExt[media], body)
		if err != nil {
			return fetchedPage{}, fmt.Errorf("web_fetch: parse %s: %w", media, err)
		}
		return fetchedPage{text: text, note: note}, nil
	case strings.HasPrefix(media, "text/") || strings.HasSuffix(media, "+json") || strings.HasSuffix(media, "+xml") || media == "application/xml":
		return fetchedPage{text: strings.ToValidUTF8(string(body), "")}, nil
	default:
		return fetchedPage{}, fmt.Errorf("web_fetch: %s content is not readable as text", media)
	}
}

// window returns the slice of text starting at offset (in characters),
// at most size long, the offset of the next slice (0 when this is the
// last) and the text's total length.
func window(text string, offset, size int) (string, int, int) {
	runes := []rune(text)
	total := len(runes)
	if offset >= total {
		return "", 0, total
	}
	end := offset + size
	if end >= total {
		return string(runes[offset:]), 0, total
	}
	return string(runes[offset:end]), end, total
}

// nonPublicNets lists ranges that net.IP's predicates miss: "this network"
// (0.0.0.0/8), carrier-grade NAT (100.64.0.0/10), where cloud providers keep
// internal services, and the deprecated IPv6 site-local range (fec0::/10).
type ipNets []*net.IPNet

var nonPublicNets = func() ipNets {
	var nets ipNets
	for _, cidr := range []string{"0.0.0.0/8", "100.64.0.0/10", "fec0::/10"} {
		_, n, _ := net.ParseCIDR(cidr)
		nets = append(nets, n)
	}
	return nets
}()

func (n ipNets) contains(ip net.IP) bool {
	for _, block := range n {
		if block.Contains(ip) {
			return true
		}
	}
	return false
}

var (
	_, nat64Net, _     = net.ParseCIDR("64:ff9b::/96")
	_, sixToFourNet, _ = net.ParseCIDR("2002::/16")
)

// embeddedIPv4 returns the IPv4 address that a NAT64 (64:ff9b::/96) or 6to4
// (2002::/16) address stands for, or nil. The gateway that routes such an
// address connects to that IPv4 target, so the target is what gets checked.
// Blocking the prefixes outright would also refuse public IPv4 sites, which
// DNS64 synthesises under the NAT64 prefix on IPv6-only networks.
func embeddedIPv4(ip net.IP) net.IP {
	ip16 := ip.To16()
	if ip.To4() != nil || ip16 == nil {
		return nil
	}
	switch {
	case nat64Net.Contains(ip):
		return net.IP(ip16[12:16])
	case sixToFourNet.Contains(ip):
		return net.IP(ip16[2:6])
	}
	return nil
}

// guardPrivateAddress refuses connections to anything that is not a
// public address — the anti-SSRF core of the tool.
func guardPrivateAddress(address string) error {
	host, _, err := net.SplitHostPort(address)
	if err != nil {
		host = address
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return nil // non-IP (unix sockets etc.) — let the dialer decide
	}
	if v4 := embeddedIPv4(ip); v4 != nil {
		ip = v4
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsMulticast() || ip.IsUnspecified() || nonPublicNets.contains(ip) {
		return fmt.Errorf("web_fetch: refusing non-public address %s", host)
	}
	return nil
}
