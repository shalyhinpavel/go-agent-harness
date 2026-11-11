package tools

import (
	"archive/zip"
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/xuri/excelize/v2"

	"github.com/shalyhinpavel/go-agent-harness/agent"
)

func toolExec(t *testing.T, tool *agent.Tool, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
	t.Helper()
	return tool.Execute(context.Background(), tc, args)
}

func TestFiles_ParseXLSX(t *testing.T) {
	f := excelize.NewFile()
	if err := f.SetSheetRow("Sheet1", "A1", &[]any{"name", "score"}); err != nil {
		t.Fatal(err)
	}
	if err := f.SetSheetRow("Sheet1", "A2", &[]any{"alice", 42}); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := f.Write(&buf); err != nil {
		t.Fatal(err)
	}

	content, note, err := ParseDocument(".xlsx", buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "name | score") || !strings.Contains(content, "alice | 42") {
		t.Fatalf("xlsx not parsed: %q", content)
	}
	if note == "" {
		t.Fatal("xlsx parse note missing")
	}
}

func TestFiles_ParsePDF(t *testing.T) {
	raw := buildTestPDF(t, "Hello PDF text layer")

	content, _, err := ParseDocument(".pdf", raw)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "Hello PDF text layer") {
		t.Fatalf("pdf text layer not extracted: %q", content)
	}
}

// buildTestPDF assembles a minimal single-page PDF with one text line —
// the pure-Go extractor has no writer, so tests carry their own fixture.
func buildTestPDF(t *testing.T, text string) []byte {
	t.Helper()

	stream := "BT /F1 24 Tf 50 700 Td (" + text + ") Tj ET"
	objects := []string{
		"<< /Type /Catalog /Pages 2 0 R >>",
		"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
		"<< /Length " + fmt.Sprint(len(stream)) + " >>\nstream\n" + stream + "\nendstream",
		"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
	}

	var buf bytes.Buffer
	buf.WriteString("%PDF-1.4\n")
	offsets := make([]int, len(objects))
	for i, obj := range objects {
		offsets[i] = buf.Len()
		fmt.Fprintf(&buf, "%d 0 obj\n%s\nendobj\n", i+1, obj)
	}

	xref := buf.Len()
	fmt.Fprintf(&buf, "xref\n0 %d\n0000000000 65535 f \n", len(objects)+1)
	for _, off := range offsets {
		fmt.Fprintf(&buf, "%010d 00000 n \n", off)
	}
	fmt.Fprintf(&buf, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(objects)+1, xref)
	return buf.Bytes()
}

func TestFiles_ParseDOCX(t *testing.T) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	fw, _ := zw.Create("word/document.xml")
	fw.Write([]byte(`<w:document><w:body><w:p><w:r><w:t>Hello</w:t></w:r></w:p><w:p><w:r><w:t t="1">docx &amp; world</w:t></w:r></w:p></w:body></w:document>`))
	zw.Close()

	content, _, err := ParseDocument(".docx", buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "Hello") || !strings.Contains(content, "docx & world") {
		t.Fatalf("docx not parsed: %q", content)
	}
}

func TestFiles_ParseCSVAndJSON(t *testing.T) {
	csvContent, _, err := ParseDocument(".csv", []byte("a,b\n\"x,y\",2\n"))
	if err != nil || csvContent != "a | b\nx,y | 2\n" {
		t.Fatalf("csv parse: %q err=%v", csvContent, err)
	}

	jsonContent, _, err := ParseDocument(".json", []byte(`{"k": [1,2]}`))
	if err != nil || !strings.Contains(jsonContent, "\"k\"") {
		t.Fatalf("json parse: %q err=%v", jsonContent, err)
	}

	// Broken JSON falls back to raw text, not an error.
	broken, _, err := ParseDocument(".json", []byte(`{oops`))
	if err != nil || broken != "{oops" {
		t.Fatalf("broken json should pass through: %q err=%v", broken, err)
	}
}

func TestShell(t *testing.T) {
	root := t.TempDir()
	sh := NewShell(ShellConfig{WorkspaceRoot: root, TimeoutS: 5})

	res, err := toolExec(t, sh, &agent.ToolContext{SessionID: "s1"}, map[string]any{"command": "echo hello && pwd"})
	if err != nil {
		t.Fatal(err)
	}
	data := res.Data.(map[string]any)
	if data["exit_code"] != 0 || !strings.Contains(data["output"].(string), "hello") {
		t.Fatalf("shell result wrong: %+v", data)
	}

	// Non-zero exit code is a result, not an error.
	res, err = toolExec(t, sh, &agent.ToolContext{SessionID: "s1"}, map[string]any{"command": "exit 3"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Data.(map[string]any)["exit_code"] != 3 {
		t.Fatalf("exit code lost: %+v", res.Data)
	}

	// Timeout is reported, not hung.
	res, err = toolExec(t, NewShell(ShellConfig{WorkspaceRoot: root, TimeoutS: 1}),
		&agent.ToolContext{SessionID: "s1"}, map[string]any{"command": "sleep 5"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Data.(map[string]any)["error"] == "" {
		t.Fatalf("timeout not reported: %+v", res.Data)
	}
}

func TestShell_EnvironmentIsScrubbed(t *testing.T) {
	// A prompt-injected `env` must not find the harness's secrets.
	t.Setenv("OPENROUTER_API_KEY", "sekret-123")
	root := t.TempDir()
	sh := NewShell(ShellConfig{WorkspaceRoot: root, TimeoutS: 5})
	res, err := toolExec(t, sh, &agent.ToolContext{SessionID: "s1"}, map[string]any{"command": "env"})
	if err != nil {
		t.Fatal(err)
	}
	out := res.Data.(map[string]any)["output"].(string)
	if strings.Contains(out, "sekret-123") {
		t.Fatal("the shell inherited an API key")
	}
	if !strings.Contains(out, "PATH=") || !strings.Contains(out, "HOME="+filepath.Join(root, "anonymous", "s1")) {
		t.Fatalf("PATH or HOME missing:\n%s", out)
	}
}

func TestShell_BackgroundChildNeitherHangsNorSurvives(t *testing.T) {
	// A background child holds the output pipe: the command still returns
	// promptly, and the child dies with it instead of running on.
	root := t.TempDir()
	sh := NewShell(ShellConfig{WorkspaceRoot: root, TimeoutS: 30})
	start := time.Now()
	res, err := toolExec(t, sh, &agent.ToolContext{SessionID: "s1"},
		map[string]any{"command": "(sleep 1; echo alive > marker; sleep 20) & echo started"})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Fatalf("command hung on its background child: %v", elapsed)
	}
	if !strings.Contains(res.Data.(map[string]any)["output"].(string), "started") {
		t.Fatalf("output lost: %+v", res.Data)
	}
	time.Sleep(1500 * time.Millisecond)
	if _, err := os.Stat(filepath.Join(root, "anonymous", "s1", "marker")); err == nil {
		t.Fatal("the background child outlived its command")
	}
}

func TestShell_OutputIsBoundedWhileItStreams(t *testing.T) {
	root := t.TempDir()
	sh := NewShell(ShellConfig{WorkspaceRoot: root, TimeoutS: 10})
	res, err := toolExec(t, sh, &agent.ToolContext{SessionID: "s1"},
		map[string]any{"command": "head -c 300000 /dev/zero | tr '\\0' a"})
	if err != nil {
		t.Fatal(err)
	}
	data := res.Data.(map[string]any)
	if len(data["output"].(string)) != maxShellOutput || data["truncated"] != true || data["dropped_bytes"] != 300000-maxShellOutput {
		t.Fatalf("output len %d, truncated %v, dropped %v", len(data["output"].(string)), data["truncated"], data["dropped_bytes"])
	}
}
