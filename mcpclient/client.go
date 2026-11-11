// Package mcpclient connects the harness to Model Context Protocol servers
// and exposes their tools to the agent as ordinary registry tools.
//
// A server is reached over streamable HTTP, legacy SSE or a stdio
// subprocess. Its tool list is read once at connect time; each tool keeps
// the server's name, description and JSON input schema, and a call is
// forwarded as tools/call with the model's arguments. Text content comes
// back to the model as text; structured content, when the server sends
// it, as JSON. A result flagged isError reaches the model as a tool error.
package mcpclient

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// Transports a server can use.
const (
	TransportStreamableHTTP = "streamable-http"
	TransportSSE            = "sse"
	TransportStdio          = "stdio"
)

// ServerConfig describes one MCP server.
type ServerConfig struct {
	// Name labels the server in logs and errors.
	Name string `yaml:"name"`
	// Transport: streamable-http (default when URL is set) | sse | stdio.
	Transport string `yaml:"transport"`
	// URL is the endpoint of an HTTP transport.
	URL string `yaml:"url"`
	// Command and Args start a stdio server. The server sees a minimal
	// environment (PATH, HOME, locale, temp dir, CA bundles) plus Env:
	// the agent's own API keys never reach a third-party subprocess unless
	// listed here.
	Command string            `yaml:"command"`
	Args    []string          `yaml:"args"`
	Env     map[string]string `yaml:"env"`
	// ToolPrefix is prepended to every tool name ("" = the server's names
	// as they are); set it when two servers expose the same name.
	ToolPrefix string `yaml:"tool_prefix"`
	// Tools restricts registration to these server tool names (empty = all).
	Tools []string `yaml:"tools"`
	// ReadOnly names tools to treat as side-effect free (safe to
	// run without approval) in addition to those the server annotates
	// with readOnlyHint.
	ReadOnly []string `yaml:"read_only"`
	// CallTimeoutS bounds one tool call (0 = 120 s).
	CallTimeoutS int `yaml:"call_timeout_s"`
}

// Server is a live connection to one MCP server.
type Server struct {
	cfg     ServerConfig
	session *mcp.ClientSession
}

// Connect opens the session (the MCP initialize handshake included).
func Connect(ctx context.Context, cfg ServerConfig) (*Server, error) {
	transport, err := newTransport(cfg)
	if err != nil {
		return nil, err
	}
	client := mcp.NewClient(&mcp.Implementation{Name: "go-agent-harness", Version: "1"}, nil)
	session, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcpclient: connect %s: %w", cfg.Name, err)
	}
	return &Server{cfg: cfg, session: session}, nil
}

func newTransport(cfg ServerConfig) (mcp.Transport, error) {
	kind := cfg.Transport
	if kind == "" {
		if cfg.Command != "" {
			kind = TransportStdio
		} else {
			kind = TransportStreamableHTTP
		}
	}
	switch kind {
	case TransportStreamableHTTP:
		if cfg.URL == "" {
			return nil, fmt.Errorf("mcpclient: server %s: url is required for %s", cfg.Name, kind)
		}
		return &mcp.StreamableClientTransport{Endpoint: cfg.URL}, nil
	case TransportSSE:
		if cfg.URL == "" {
			return nil, fmt.Errorf("mcpclient: server %s: url is required for %s", cfg.Name, kind)
		}
		return &mcp.SSEClientTransport{Endpoint: cfg.URL}, nil
	case TransportStdio:
		if cfg.Command == "" {
			return nil, fmt.Errorf("mcpclient: server %s: command is required for stdio", cfg.Name)
		}
		cmd := exec.Command(cfg.Command, cfg.Args...)
		cmd.Env = stdioEnv(os.Environ(), cfg.Env)
		cmd.Stderr = os.Stderr
		return &mcp.CommandTransport{Command: cmd}, nil
	}
	return nil, fmt.Errorf("mcpclient: server %s: unknown transport %q", cfg.Name, kind)
}

// inheritedEnv names the variables a stdio server gets from the agent's
// environment; LC_* is matched by prefix.
var inheritedEnv = map[string]bool{
	"PATH": true, "HOME": true, "USER": true, "LOGNAME": true, "SHELL": true,
	"TMPDIR": true, "TMP": true, "TEMP": true, "LANG": true, "TZ": true,
	"SSL_CERT_FILE": true, "SSL_CERT_DIR": true, "NODE_EXTRA_CA_CERTS": true,
	// Windows needs these to start any process at all.
	"SYSTEMROOT": true, "SYSTEMDRIVE": true, "COMSPEC": true, "PATHEXT": true,
	"USERPROFILE": true, "APPDATA": true, "LOCALAPPDATA": true,
}

// stdioEnv builds the environment of a stdio server: the safe subset of
// parent, then extra on top (extra wins).
func stdioEnv(parent []string, extra map[string]string) []string {
	var env []string
	for _, kv := range parent {
		k, _, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		if inheritedEnv[strings.ToUpper(k)] || strings.HasPrefix(k, "LC_") {
			if _, overridden := extra[k]; !overridden {
				env = append(env, kv)
			}
		}
	}
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		env = append(env, k+"="+extra[k])
	}
	return env
}

// Close ends the session (and stops a stdio subprocess).
func (s *Server) Close() error { return s.session.Close() }

// Tools lists the server's tools as registry tools.
func (s *Server) Tools(ctx context.Context) ([]*agent.Tool, error) {
	want := set(s.cfg.Tools)
	readOnly := set(s.cfg.ReadOnly)
	var out []*agent.Tool
	for t, err := range s.session.Tools(ctx, nil) {
		if err != nil {
			return nil, fmt.Errorf("mcpclient: list tools of %s: %w", s.cfg.Name, err)
		}
		if len(want) > 0 && !want[t.Name] {
			continue
		}
		schema, err := toSchema(t.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("mcpclient: %s tool %s: %w", s.cfg.Name, t.Name, err)
		}
		out = append(out, &agent.Tool{
			Name:        s.cfg.ToolPrefix + t.Name,
			Description: t.Description,
			Parameters:  schema,
			Execute:     s.caller(t.Name),
			ReadOnly:    readOnly[t.Name] || (t.Annotations != nil && t.Annotations.ReadOnlyHint),
		})
	}
	return out, nil
}

// caller forwards one tool to the server.
func (s *Server) caller(name string) agent.ToolExecuteFunc {
	timeout := time.Duration(s.cfg.CallTimeoutS) * time.Second
	if timeout <= 0 {
		timeout = 120 * time.Second
	}
	return func(ctx context.Context, _ *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
		ctx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		if args == nil {
			args = map[string]any{}
		}
		res, err := s.session.CallTool(ctx, &mcp.CallToolParams{Name: name, Arguments: args})
		if err != nil {
			return nil, fmt.Errorf("mcpclient: %s/%s: %w", s.cfg.Name, name, err)
		}
		text := contentText(res.Content)
		if res.IsError {
			if text == "" {
				text = "tool reported an error"
			}
			return &agent.ToolResult{Error: text}, nil
		}
		// The spec has servers mirror structured output in a text block, and
		// text is what a model reads best: a string return comes back from
		// FastMCP as {"result": "..."}, which as JSON escapes every newline.
		// Structured content is used only when no text came with it.
		if text == "" && res.StructuredContent != nil {
			return &agent.ToolResult{Data: res.StructuredContent}, nil
		}
		return &agent.ToolResult{Data: text}, nil
	}
}

// contentText joins the text blocks of a result; non-text blocks are
// named so the model knows they were there.
func contentText(blocks []mcp.Content) string {
	var parts []string
	for _, c := range blocks {
		switch v := c.(type) {
		case *mcp.TextContent:
			parts = append(parts, v.Text)
		case *mcp.ImageContent:
			parts = append(parts, "[image "+v.MIMEType+" omitted]")
		case *mcp.AudioContent:
			parts = append(parts, "[audio "+v.MIMEType+" omitted]")
		default:
			if raw, err := json.Marshal(c); err == nil {
				parts = append(parts, string(raw))
			}
		}
	}
	return strings.Join(parts, "\n")
}

// toSchema converts the server's input schema (any JSON value) into the
// neutral schema map; a missing schema is an empty object.
func toSchema(in any) (llm.Schema, error) {
	if in == nil {
		return llm.Schema{"type": "object", "properties": map[string]any{}}, nil
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}
	var s llm.Schema
	if err := json.Unmarshal(raw, &s); err != nil {
		return nil, err
	}
	if _, ok := s["type"]; !ok {
		s["type"] = "object"
	}
	return s, nil
}

func set(names []string) map[string]bool {
	m := make(map[string]bool, len(names))
	for _, n := range names {
		m[n] = true
	}
	return m
}

// Pool holds the connections opened for one harness.
type Pool struct{ servers []*Server }

// ConnectAll connects every server and registers its tools; on any
// failure the connections opened so far are closed. Name clashes with
// tools already in the registry are errors (use ToolPrefix).
func ConnectAll(ctx context.Context, cfgs []ServerConfig, reg *agent.ToolRegistry) (*Pool, error) {
	p := &Pool{}
	for _, cfg := range cfgs {
		srv, err := Connect(ctx, cfg)
		if err != nil {
			p.Close()
			return nil, err
		}
		p.servers = append(p.servers, srv)
		tools, err := srv.Tools(ctx)
		if err != nil {
			p.Close()
			return nil, err
		}
		for _, t := range tools {
			if err := reg.Register(t); err != nil {
				p.Close()
				return nil, fmt.Errorf("mcpclient: register %s from %s: %w", t.Name, cfg.Name, err)
			}
		}
	}
	return p, nil
}

// Close closes every connection.
func (p *Pool) Close() {
	if p == nil {
		return
	}
	for _, s := range p.servers {
		_ = s.Close()
	}
	p.servers = nil
}
