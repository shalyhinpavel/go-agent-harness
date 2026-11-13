package tools

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
)

func fetch(t *testing.T, cfg WebFetchConfig, args map[string]any) (map[string]any, error) {
	t.Helper()
	res, err := toolExec(t, NewWebFetch(cfg), &agent.ToolContext{}, args)
	if err != nil {
		return nil, err
	}
	return res.Data.(map[string]any), nil
}

func TestHTMLToText_KeepsLinksAndStructure(t *testing.T) {
	base, _ := url.Parse("https://example.com/docs/index.html")
	title, text := htmlToText(`<html><head><title> Docs  home </title><style>body{}</style></head>
<body><nav><a href="/menu">Menu</a></nav>
<h1>Guide</h1><script>evil()</script>
<p>Read the <a href="intro.html">intro</a> or <a href="https://other.org/x#frag">other</a>
and <a href="#top">top</a>, <a href="mailto:a@b.c">mail</a>, <a href="javascript:void(0)">js</a>.</p>
<ul><li>one</li><li>two</li></ul><footer>footer text</footer></body></html>`, base)

	if title != "Docs home" {
		t.Fatalf("title = %q", title)
	}
	for _, want := range []string{
		"# Guide",
		"[intro](https://example.com/docs/intro.html)", // relative link made absolute
		"[other](https://other.org/x)",                 // fragment dropped
		"- one", "- two",
		"and top, mail, js.", // unfollowable links stay plain text
	} {
		if !strings.Contains(text, want) {
			t.Errorf("missing %q in:\n%s", want, text)
		}
	}
	for _, bad := range []string{"evil()", "Menu", "footer text", "<"} {
		if strings.Contains(text, bad) {
			t.Errorf("%q leaked into:\n%s", bad, text)
		}
	}
}

func TestWebFetch_ReadsPageAndFollowsRedirect(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/old", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/new", http.StatusFound) })
	mux.HandleFunc("/new", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(`<title>New</title><p>Go to <a href="/page2">page 2</a></p>`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	data, err := fetch(t, WebFetchConfig{}, map[string]any{"url": srv.URL + "/old"})
	if err != nil {
		t.Fatal(err)
	}
	if data["url"] != srv.URL+"/new" || data["title"] != "New" {
		t.Fatalf("url/title = %v / %v", data["url"], data["title"])
	}
	if c := data["content"].(string); !strings.Contains(c, "[page 2]("+srv.URL+"/page2)") {
		t.Fatalf("link not kept: %q", c)
	}
}

func TestWebFetch_PagesLongTextWithOffset(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(strings.Repeat("абвгд", 100))) // 500 characters, multi-byte
	}))
	defer srv.Close()

	cfg := WebFetchConfig{PageChars: 200}
	first, err := fetch(t, cfg, map[string]any{"url": srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if first["total_chars"] != 500 || first["next_offset"] != 200 || len([]rune(first["content"].(string))) != 200 {
		t.Fatalf("first page: %v", first)
	}
	last, err := fetch(t, cfg, map[string]any{"url": srv.URL, "offset": float64(400)})
	if err != nil {
		t.Fatal(err)
	}
	if _, more := last["next_offset"]; more || len([]rune(last["content"].(string))) != 100 {
		t.Fatalf("last page: %v", last)
	}
}

func TestWebFetch_StatusErrorsCarryAdvice(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/missing":
			http.NotFound(w, r)
		case "/blocked":
			w.WriteHeader(http.StatusForbidden)
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	defer srv.Close()

	for path, want := range map[string]string{"/missing": "not found", "/blocked": "refuses automated access", "/boom": "HTTP 500"} {
		if _, err := fetch(t, WebFetchConfig{}, map[string]any{"url": srv.URL + path}); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: error = %v, want mention of %q", path, err, want)
		}
	}
}

func TestWebFetch_RefusesBinaryAndBadSchemes(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
	}))
	defer srv.Close()

	if _, err := fetch(t, WebFetchConfig{}, map[string]any{"url": srv.URL}); err == nil || !strings.Contains(err.Error(), "not readable") {
		t.Fatalf("binary content: %v", err)
	}
	if _, err := fetch(t, WebFetchConfig{}, map[string]any{"url": "ftp://example.com"}); err == nil {
		t.Fatal("non-http scheme must be refused")
	}
}

func TestWebFetch_PrivateAddressBlocked(t *testing.T) {
	// httptest binds to loopback: with BlockPrivate the tool must refuse.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("secret"))
	}))
	defer srv.Close()

	if _, err := fetch(t, WebFetchConfig{BlockPrivate: true}, map[string]any{"url": srv.URL}); err == nil {
		t.Fatal("loopback fetch must be refused by the anti-SSRF guard")
	}
}

func TestGuardPrivateAddress(t *testing.T) {
	for _, addr := range []string{
		"127.0.0.1:80", "10.0.0.5:80", "192.168.1.1:443", "169.254.169.254:80",
		"100.64.0.1:80", "100.127.255.254:80", "0.1.2.3:80", "224.0.0.1:80",
		"[::1]:80", "[fe80::1]:80", "[ff02::1]:80", "[::ffff:10.0.0.1]:80",
		// NAT64 and 6to4 carry an IPv4 target that the gateway reaches for us:
		// 127.0.0.1 and 169.254.169.254 inside each. Site-local is deprecated
		// but still routable on old networks.
		"[64:ff9b::7f00:1]:443", "[64:ff9b::a9fe:a9fe]:80", "[2002:7f00:1::1]:80", "[2002:a9fe:a9fe::1]:80",
		"[fec0::1]:80",
	} {
		if err := guardPrivateAddress(addr); err == nil {
			t.Errorf("%s must be refused", addr)
		}
	}
	// Public IPv4 reached through NAT64 (DNS64 synthesises these on IPv6-only
	// networks) and through 6to4 must keep working.
	for _, addr := range []string{"93.184.216.34:443", "100.63.255.255:80", "100.128.0.1:80", "[2606:4700::1111]:443", "[64:ff9b::808:808]:443", "[2002:808:808::1]:443"} {
		if err := guardPrivateAddress(addr); err != nil {
			t.Errorf("%s must pass: %v", addr, err)
		}
	}
}
