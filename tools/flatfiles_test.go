package tools

import (
	"fmt"
	"strings"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
)

func TestFlatFileTools_Roundtrip(t *testing.T) {
	root := t.TempDir()
	tc := &agent.ToolContext{UserID: "u1", SessionID: "s1"}
	write, read, list := NewWriteFile(root), NewReadFile(root), NewListFiles(root)

	if write.ReadOnly || !read.ReadOnly || !list.ReadOnly {
		t.Fatalf("ReadOnly flags: write=%v read=%v list=%v", write.ReadOnly, read.ReadOnly, list.ReadOnly)
	}
	if _, err := toolExec(t, write, tc, map[string]any{"path": "plan/model.csv", "content": "month,cash\n1,100\n"}); err != nil {
		t.Fatal(err)
	}
	res, err := toolExec(t, read, tc, map[string]any{"path": "plan/model.csv"})
	if err != nil {
		t.Fatal(err)
	}
	if c := res.Data.(map[string]any)["content"].(string); !strings.Contains(c, "cash") {
		t.Fatalf("read content = %q", c)
	}
	res, err = toolExec(t, list, tc, map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	// list_files shows nested files by relative path, so the model can
	// open what it wrote into a subdirectory.
	if res.Data.(map[string]any)["count"] != 2 || !strings.Contains(fmt.Sprint(res.Data), "plan/model.csv") {
		t.Fatalf("list = %+v", res.Data)
	}

	// No path escape, sessions and users isolated even when sharing a session id.
	if _, err := toolExec(t, read, tc, map[string]any{"path": "../../etc/passwd"}); err == nil {
		t.Fatal("path escape must fail")
	}
	if _, err := toolExec(t, read, &agent.ToolContext{UserID: "u2", SessionID: "other"}, map[string]any{"path": "plan/model.csv"}); err == nil {
		t.Fatal("session isolation broken")
	}
	if _, err := toolExec(t, read, &agent.ToolContext{UserID: "u2", SessionID: "s1"}, map[string]any{"path": "plan/model.csv"}); err == nil {
		t.Fatal("same session id under another user must not share the workspace")
	}
}

func TestFlatFileTools_Schemas(t *testing.T) {
	// No `op` switch: the point of the flat tools.
	for _, tool := range []*agent.Tool{NewReadFile("w"), NewWriteFile("w"), NewListFiles("w")} {
		props, _ := tool.Parameters["properties"].(map[string]any)
		if _, ok := props["op"]; ok {
			t.Fatalf("%s must not have an op parameter", tool.Name)
		}
	}
}

func TestBashRegistersByName(t *testing.T) {
	registry := agent.NewToolRegistry()
	if err := RegisterSelected(registry, []string{"bash"}, Options{FilesDir: t.TempDir(), ShellTimeoutS: 5}); err != nil {
		t.Fatal(err)
	}
	bash, ok := registry.Get("bash")
	if !ok {
		t.Fatal("bash not registered")
	}
	res, err := toolExec(t, bash, &agent.ToolContext{SessionID: "s1"}, map[string]any{"command": "echo hi"})
	if err != nil || !strings.Contains(res.Data.(map[string]any)["output"].(string), "hi") {
		t.Fatalf("bash result: %v %v", res, err)
	}
}

func TestHasFileTool(t *testing.T) {
	if !HasFileTool([]string{"web_fetch", "list_files"}) || HasFileTool([]string{"web_fetch", "bash"}) {
		t.Fatal("HasFileTool wrong")
	}
}
