package tools

import (
	"net/url"
	"strings"

	"golang.org/x/net/html"
)

// htmlToText is the page extraction behind web_fetch: a title plus the
// visible text, block structure kept as line breaks, headings as "#",
// list items as "-", and links as [text](absolute URL) so the model can
// follow them. Scripts, styles and page furniture (nav, footer, aside) are
// dropped. It is intentionally simple — good enough to read most pages,
// easy to replace with a real readability library.
func htmlToText(src string, base *url.URL) (title, text string) {
	doc, err := html.Parse(strings.NewReader(src))
	if err != nil {
		return "", strings.TrimSpace(src)
	}
	w := textWriter{base: base}
	w.walk(doc)
	title = strings.TrimSpace(strings.Join(strings.Fields(w.title), " "))
	return title, w.String()
}

// skipTags are elements whose subtree never reaches the model.
var skipTags = map[string]bool{
	"script": true, "style": true, "noscript": true, "template": true,
	"svg": true, "iframe": true, "head": true,
	"nav": true, "footer": true, "aside": true,
}

// blockTags start and end a line.
var blockTags = map[string]bool{
	"p": true, "div": true, "section": true, "article": true, "main": true,
	"header": true, "ul": true, "ol": true, "li": true, "table": true, "tr": true,
	"blockquote": true, "pre": true, "br": true, "hr": true,
	"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true,
	"dl": true, "dt": true, "dd": true, "figure": true, "figcaption": true,
	"form": true, "details": true, "summary": true,
}

type textWriter struct {
	base  *url.URL
	sb    strings.Builder
	title string
}

// String collapses whitespace per line and runs of blank lines.
func (w *textWriter) String() string {
	var out []string
	blank := false
	for _, line := range strings.Split(w.sb.String(), "\n") {
		line = strings.Join(strings.Fields(line), " ")
		if line == "" {
			if !blank && len(out) > 0 {
				out = append(out, "")
			}
			blank = true
			continue
		}
		blank = false
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}

func (w *textWriter) walk(n *html.Node) {
	switch n.Type {
	case html.TextNode:
		w.sb.WriteString(n.Data)
		return
	case html.ElementNode:
		if n.Data == "title" {
			w.title = textOf(n)
			return
		}
		if skipTags[n.Data] || hiddenOrChrome(n) {
			if n.Data == "head" {
				w.findTitle(n)
			}
			return
		}
		if n.Data == "a" && w.link(n) {
			return
		}
		block := blockTags[n.Data]
		if block {
			w.sb.WriteString("\n")
			switch n.Data {
			case "h1", "h2", "h3", "h4", "h5", "h6":
				w.sb.WriteString(strings.Repeat("#", int(n.Data[1]-'0')) + " ")
			case "li", "dt":
				w.sb.WriteString("- ")
			}
		}
		if n.Data == "td" || n.Data == "th" {
			w.sb.WriteString(" | ")
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			w.walk(c)
		}
		if block {
			w.sb.WriteString("\n")
		}
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		w.walk(c)
	}
}

// hiddenOrChrome reports elements a reader would not see or would not call
// content: hidden ones, "skip to content" links and navigation landmarks.
func hiddenOrChrome(n *html.Node) bool {
	for _, a := range n.Attr {
		switch a.Key {
		case "hidden":
			return true
		case "aria-hidden":
			if a.Val == "true" {
				return true
			}
		case "role":
			if a.Val == "navigation" || a.Val == "banner" || a.Val == "search" {
				return true
			}
		case "class":
			for _, c := range strings.Fields(a.Val) {
				if c == "sr-only" || c == "visually-hidden" || c == "skip-link" || c == "d-none" {
					return true
				}
			}
		}
	}
	return false
}

// findTitle records the page title from inside a skipped <head>.
func (w *textWriter) findTitle(n *html.Node) {
	if n.Type == html.ElementNode && n.Data == "title" {
		w.title = textOf(n)
		return
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		w.findTitle(c)
	}
}

// link writes [text](url) for an anchor with a followable href and reports
// whether it handled the node. Fragment-only, javascript:, mailto: and
// empty links stay plain text.
func (w *textWriter) link(n *html.Node) bool {
	var href string
	for _, a := range n.Attr {
		if a.Key == "href" {
			href = strings.TrimSpace(a.Val)
		}
	}
	label := strings.Join(strings.Fields(textOf(n)), " ")
	if href == "" || strings.HasPrefix(href, "#") || label == "" {
		return false
	}
	target, err := url.Parse(href)
	if err != nil {
		return false
	}
	if w.base != nil {
		target = w.base.ResolveReference(target)
	}
	if target.Scheme != "http" && target.Scheme != "https" {
		return false
	}
	target.Fragment = ""
	w.sb.WriteString("[" + label + "](" + target.String() + ")")
	return true
}

// textOf returns the concatenated visible text of a node's subtree.
func textOf(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}
	if n.Type == html.ElementNode && skipTags[n.Data] {
		return ""
	}
	var sb strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		sb.WriteString(textOf(c))
	}
	return sb.String()
}
