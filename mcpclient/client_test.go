package mcpclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// testServer serves a small stateful MCP server over streamable HTTP
// and records the order in which tools were called.
func testServer(t *testing.T) (url string, calls func() []string) {
	t.Helper()
	var mu sync.Mutex
	var seen []string
	record := func(name string) {
		mu.Lock()
		seen = append(seen, name)
		mu.Unlock()
	}
	srv := mcp.NewServer(&mcp.Implementation{Name: "sim", Version: "0"}, nil)
	srv.AddTool(&mcp.Tool{
		Name:        "get_status",
		Description: "Day, cash and alerts.",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{}},
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true},
	}, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		record("get_status")
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "day 1, cash 1500"}}}, nil
	})
	srv.AddTool(&mcp.Tool{
		Name:        "order_goods",
		Description: "Order boxes from a supplier.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"sku": map[string]any{"type": "string"}, "boxes": map[string]any{"type": "integer"}},
			"required":   []string{"sku", "boxes"},
		},
	}, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		record("order_goods")
		var args struct {
			SKU   string `json:"sku"`
			Boxes int    `json:"boxes"`
		}
		if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
			return nil, err
		}
		if args.Boxes <= 0 {
			return &mcp.CallToolResult{IsError: true, Content: []mcp.Content{&mcp.TextContent{Text: "boxes must be positive"}}}, nil
		}
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "ordered " + args.SKU}}}, nil
	})
	srv.AddTool(&mcp.Tool{
		Name:        "read_notes",
		Description: "Read saved notes.",
		InputSchema: map[string]any{"type": "object"},
	}, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		record("read_notes")
		// FastMCP's shape for a string return: the text plus the same
		// value wrapped as structured content.
		return &mcp.CallToolResult{
			Content:           []mcp.Content{&mcp.TextContent{Text: "none"}},
			StructuredContent: map[string]any{"result": "none"},
		}, nil
	})
	srv.AddTool(&mcp.Tool{
		Name:        "stock",
		Description: "Units per product, structured only.",
		InputSchema: map[string]any{"type": "object"},
	}, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		record("stock")
		return &mcp.CallToolResult{StructuredContent: map[string]any{"cola": 3}}, nil
	})
	h := mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil)
	ts := httptest.NewServer(h)
	t.Cleanup(ts.Close)
	return ts.URL, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), seen...)
	}
}

func TestConnectAll_RegistersToolsAndForwardsCalls(t *testing.T) {
	url, _ := testServer(t)
	reg := agent.NewToolRegistry()
	pool, err := ConnectAll(context.Background(), []ServerConfig{{
		Name: "sim", URL: url, ReadOnly: []string{"read_notes"},
	}}, reg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	status, ok := reg.Get("get_status")
	if !ok || !status.ReadOnly {
		t.Fatal("get_status must be registered and read-only (server annotation)")
	}
	notes, _ := reg.Get("read_notes")
	if notes == nil || !notes.ReadOnly {
		t.Fatal("read_notes must be read-only (config override)")
	}
	order, _ := reg.Get("order_goods")
	if order == nil || order.ReadOnly {
		t.Fatal("order_goods must be registered and stay with the lead")
	}
	if req, _ := order.Parameters["required"].([]any); len(req) != 2 {
		t.Fatalf("input schema not carried over: %v", order.Parameters)
	}

	res, err := status.Execute(context.Background(), &agent.ToolContext{}, nil)
	if err != nil || res.Data != "day 1, cash 1500" {
		t.Fatalf("get_status = %+v, %v", res, err)
	}
	res, err = order.Execute(context.Background(), &agent.ToolContext{}, map[string]any{"sku": "cola", "boxes": 2})
	if err != nil || res.Data != "ordered cola" {
		t.Fatalf("order_goods = %+v, %v", res, err)
	}
	// Text wins over its structured mirror; structured content alone is
	// passed through as data.
	res, err = notes.Execute(context.Background(), &agent.ToolContext{}, nil)
	if err != nil || res.Data != "none" {
		t.Fatalf("read_notes = %+v, %v; want the text, not the wrapper", res, err)
	}
	stock, _ := reg.Get("stock")
	res, err = stock.Execute(context.Background(), &agent.ToolContext{}, nil)
	if m, ok := res.Data.(map[string]any); err != nil || !ok || m["cola"] != float64(3) {
		t.Fatalf("stock = %+v, %v; want the structured content", res, err)
	}
	// A tool-level error reaches the model as a tool error, not a failure.
	res, err = order.Execute(context.Background(), &agent.ToolContext{}, map[string]any{"sku": "cola", "boxes": 0})
	if err != nil || !strings.Contains(res.Error, "positive") {
		t.Fatalf("isError result = %+v, %v", res, err)
	}
}

func TestConnectAll_PrefixFilterAndClash(t *testing.T) {
	url, _ := testServer(t)
	reg := agent.NewToolRegistry()
	pool, err := ConnectAll(context.Background(), []ServerConfig{{
		Name: "sim", URL: url, ToolPrefix: "vend_", Tools: []string{"get_status"},
	}}, reg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()
	if _, ok := reg.Get("vend_get_status"); !ok {
		t.Fatal("prefixed tool missing")
	}
	if len(reg.List()) != 1 {
		t.Fatalf("filter must register only get_status, got %d tools", len(reg.List()))
	}
	// The same names again without a prefix clash with nothing; with the
	// same prefix they clash and the connection is closed.
	if _, err := ConnectAll(context.Background(), []ServerConfig{{Name: "sim2", URL: url, ToolPrefix: "vend_"}}, reg); err == nil {
		t.Fatal("a name clash must fail")
	}
}

func TestConnect_BadConfig(t *testing.T) {
	for _, cfg := range []ServerConfig{
		{Name: "a"},                       // no url, no command
		{Name: "b", Transport: "sse"},     // sse without url
		{Name: "c", Transport: "stdio"},   // stdio without command
		{Name: "d", Transport: "carrier"}, // unknown
	} {
		if _, err := Connect(context.Background(), cfg); err == nil {
			t.Fatalf("config %+v must be rejected", cfg)
		}
	}
}

// With max_parallel_tools 1 the engine runs a batch of calls one after
// another in the order the model gave them: order-dependent environments
// (order goods, then restock) rely on it.
func TestEngine_SequentialToolsKeepModelOrder(t *testing.T) {
	url, calls := testServer(t)
	reg := agent.NewToolRegistry()
	pool, err := ConnectAll(context.Background(), []ServerConfig{{Name: "sim", URL: url}}, reg)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	batch := []llm.ToolCall{
		{ID: "1", Name: "get_status", Args: map[string]any{}},
		{ID: "2", Name: "order_goods", Args: map[string]any{"sku": "a", "boxes": 1}},
		{ID: "3", Name: "read_notes", Args: map[string]any{}},
		{ID: "4", Name: "order_goods", Args: map[string]any{"sku": "b", "boxes": 1}},
		{ID: "5", Name: "get_status", Args: map[string]any{}},
	}
	provider := llm.NewMockProvider(llm.MockTurn{ToolCalls: batch}, llm.MockTurn{Text: "done"})
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider: provider, SystemPrompt: "s", MaxIterations: 3, MaxParallelTools: 1,
	}, reg, agent.NewSessionManager())
	if _, err := engine.Chat(context.Background(), agent.ChatRequest{UserID: "u", SessionID: "s", Message: "go"}); err != nil {
		t.Fatal(err)
	}
	want := []string{"get_status", "order_goods", "read_notes", "order_goods", "get_status"}
	if got := calls(); strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("server saw %v, want %v", got, want)
	}
}

func TestStdioEnv_KeepsSecretsOut(t *testing.T) {
	parent := []string{
		"PATH=/usr/bin", "HOME=/home/u", "LC_ALL=C", "LANG=en_US.UTF-8",
		"OPENAI_API_KEY=sk-secret", "ANTHROPIC_API_KEY=sk-ant", "AWS_SECRET_ACCESS_KEY=x",
		"TZ=UTC", "HTTP_PROXY=http://user:pw@proxy",
	}
	env := stdioEnv(parent, map[string]string{"TZ": "Europe/Berlin", "TOKEN": "t"})
	joined := "\n" + strings.Join(env, "\n") + "\n"
	for _, secret := range []string{"sk-secret", "sk-ant", "AWS_SECRET", "pw@proxy"} {
		if strings.Contains(joined, secret) {
			t.Fatalf("%q leaked into the server environment: %v", secret, env)
		}
	}
	for _, want := range []string{"\nPATH=/usr/bin\n", "\nHOME=/home/u\n", "\nLC_ALL=C\n", "\nTZ=Europe/Berlin\n", "\nTOKEN=t\n"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("missing %q in %v", strings.TrimSpace(want), env)
		}
	}
	if strings.Contains(joined, "\nTZ=UTC\n") {
		t.Fatalf("Env must override the inherited value: %v", env)
	}
}
