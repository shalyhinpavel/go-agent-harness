package bootstrap

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/mcpclient"
)

func TestBuildEngine_MCPServerToolsJoinRegistry(t *testing.T) {
	srv := mcp.NewServer(&mcp.Implementation{Name: "sim", Version: "0"}, nil)
	srv.AddTool(&mcp.Tool{Name: "get_status", Description: "status", InputSchema: map[string]any{"type": "object"}},
		func(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}, nil
		})
	ts := httptest.NewServer(mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return srv }, nil))
	defer ts.Close()

	h := buildHarness(t, func(c *config.Config) {
		c.Tools.Enabled = nil
		c.Tools.MCPServers = []mcpclient.ServerConfig{{Name: "sim", URL: ts.URL}}
	})
	defer h.Close()
	if _, ok := h.Registry.Get("get_status"); !ok {
		t.Fatal("MCP tool missing from the registry")
	}
	if h.MCP == nil {
		t.Fatal("harness must keep the pool to close it")
	}
}

func TestBuildEngine_MCPServerDownFails(t *testing.T) {
	cfg := config.Default()
	cfg.Provider.Kind = "mock"
	cfg.Prompts.Dir = ""
	cfg.Tools.MCPServers = []mcpclient.ServerConfig{{Name: "gone", URL: "http://127.0.0.1:1/mcp"}}
	if _, err := BuildEngine(cfg); err == nil {
		t.Fatal("an unreachable MCP server must fail the build")
	}
}
