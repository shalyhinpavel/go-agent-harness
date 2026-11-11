package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/bootstrap"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// Outcomes of a check.
const (
	pass = "PASS"
	fail = "FAIL"
	info = "INFO" // observed behavior that is optional for the backend
)

type result struct {
	name    string
	outcome string
	detail  string
	latency time.Duration
	usage   llm.Usage
}

type check struct {
	name string
	run  func(ctx context.Context, p llm.LLMProvider) result
}

// runCheck runs the conformance matrix against the configured provider:
// every behavior the harness relies on, checked on the real backend
// instead of assumed from its docs. Wire tests pin the adapters to
// recorded shapes; this pins the backend to the adapters. It returns 1
// when any check fails; INFO rows never fail the run.
func runCheck(args []string) int {
	logLevel.Set(slog.LevelWarn)
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	configPath := configFlag(fs)
	model := fs.String("model", "", "model id (overrides the config)")
	timeout := fs.Duration("timeout", 2*time.Minute, "per-check timeout")
	only := fs.String("only", "", "run only the checks whose name contains this")
	effort := fs.String("reasoning-effort", "", "override provider.params.reasoning_effort")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: harness check [flags]\n\nRuns the provider contract against the live backend.")
		fs.PrintDefaults()
	}
	_ = fs.Parse(args)

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fail1(err)
	}
	if *model != "" {
		cfg.Provider.Model = *model
	}
	if *effort != "" {
		cfg.Provider.Params.ReasoningEffort = *effort
	}
	cfg.Provider = bootstrap.Resolve(cfg.Provider)
	if cfg.Provider.Kind == "mock" {
		return fail1(fmt.Errorf("no API key found: check needs a real backend"))
	}
	p, err := bootstrap.Provider(cfg.Provider)
	if err != nil {
		return fail1(err)
	}
	fmt.Printf("provider=%s model=“%s”\n", p.Name(), cfg.Provider.Model)
	if !cfg.Provider.Params.IsZero() {
		params, _ := json.Marshal(cfg.Provider.Params)
		fmt.Printf("params=%s\n", params)
	}
	fmt.Println()

	checks := matrix()
	results := make([]result, 0, len(checks))
	for _, c := range checks {
		if *only != "" && !strings.Contains(c.name, *only) {
			continue
		}
		fmt.Fprintf(os.Stderr, "running %s…\n", c.name)
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		start := time.Now()
		r := c.run(ctx, p)
		cancel()
		r.name, r.latency = c.name, time.Since(start)
		results = append(results, r)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "CHECK\tRESULT\tLATENCY\tTOKENS in/out\tDETAIL")
	failed := 0
	var total llm.Usage
	for _, r := range results {
		if r.outcome == fail {
			failed++
		}
		total.Add(r.usage)
		fmt.Fprintf(w, "%s\t%s\t%s\t%d/%d\t%s\n", r.name, r.outcome, r.latency.Round(time.Millisecond),
			r.usage.InputTokens, r.usage.OutputTokens, oneLine(r.detail, 110))
	}
	w.Flush()
	fmt.Printf("\n%d check(s), %d failed; tokens in/out %d/%d\n", len(results), failed, total.InputTokens, total.OutputTokens)
	if failed > 0 {
		return 1
	}
	return 0
}

// stream collects one streamed call.
type collected struct {
	text     string
	thinking bool
	calls    []llm.ToolCall
	usage    *llm.Usage
	stop     llm.StopReason
	err      error
}

func streamOnce(ctx context.Context, p llm.LLMProvider, req *llm.ChatRequest) collected {
	var c collected
	events, err := p.StreamChat(ctx, req)
	if err != nil {
		c.err = err
		return c
	}
	var sb strings.Builder
	for ev := range events {
		switch ev.Kind {
		case llm.EventText:
			sb.WriteString(ev.Text)
		case llm.EventThinking:
			c.thinking = true
		case llm.EventToolCalls:
			c.calls = append(c.calls, ev.ToolCalls...)
		case llm.EventError:
			c.err = ev.Err
		case llm.EventDone:
			c.usage, c.stop = ev.Usage, ev.StopReason
		}
	}
	c.text = sb.String()
	return c
}

func (c collected) spent() llm.Usage {
	if c.usage == nil {
		return llm.Usage{}
	}
	return *c.usage
}

var addTool = llm.ToolDef{
	Name:        "add",
	Description: "Adds two integers and returns the sum.",
	Parameters: llm.Schema{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "integer"},
		},
		"required": []string{"a", "b"},
	},
}

func user(text string) []llm.Message { return []llm.Message{*llm.UserText(text)} }

func failed(detail string, args ...any) result {
	return result{outcome: fail, detail: fmt.Sprintf(detail, args...)}
}

func matrix() []check {
	return []check{
		{"text + usage + stop", func(ctx context.Context, p llm.LLMProvider) result {
			c := streamOnce(ctx, p, &llm.ChatRequest{Messages: user("Reply with exactly the word PONG and nothing else.")})
			r := result{usage: c.spent()}
			switch {
			case c.err != nil:
				return failed("%v", c.err)
			case !strings.Contains(strings.ToUpper(c.text), "PONG"):
				r.outcome, r.detail = fail, fmt.Sprintf("text “%s”", c.text)
			case c.usage == nil || c.usage.InputTokens == 0 || c.usage.OutputTokens == 0:
				r.outcome, r.detail = fail, fmt.Sprintf("usage missing: %+v", c.usage)
			case c.stop != llm.StopEnd:
				r.outcome, r.detail = fail, fmt.Sprintf("stop reason “%s”, want end", c.stop)
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("“%s”", strings.TrimSpace(c.text))
			}
			return r
		}},
		{"generate + usage", func(ctx context.Context, p llm.LLMProvider) result {
			text, usage, err := p.Generate(ctx, &llm.ChatRequest{Messages: user("Reply with exactly: OK")})
			r := result{usage: usage}
			switch {
			case err != nil:
				return failed("%v", err)
			case usage.InputTokens == 0:
				r.outcome, r.detail = fail, "Generate reported no usage"
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("“%s”", strings.TrimSpace(text))
			}
			return r
		}},
		{"tool round-trip (required)", func(ctx context.Context, p llm.LLMProvider) result {
			req := &llm.ChatRequest{
				Messages:   user("What is 1234 + 4321? Use the add tool."),
				Tools:      []llm.ToolDef{addTool},
				ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceRequired},
			}
			first := streamOnce(ctx, p, req)
			r := result{usage: first.spent()}
			if first.err != nil {
				return failed("%v", first.err)
			}
			if len(first.calls) == 0 || first.calls[0].Name != "add" {
				return failed("no add call (text “%s”)", first.text)
			}
			if first.stop != llm.StopToolUse {
				r.detail = fmt.Sprintf("stop “%s” (want tool_use); ", first.stop)
			}
			call := first.calls[0]
			a, b := num(call.Args["a"]), num(call.Args["b"])
			req.Messages = append(req.Messages,
				llm.Message{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(call)}},
				llm.Message{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: call.ID, Name: call.Name, Content: map[string]any{"sum": a + b}})}},
			)
			req.ToolChoice = llm.ToolChoice{}
			second := streamOnce(ctx, p, req)
			r.usage.Add(second.spent())
			switch {
			case second.err != nil:
				return failed("after tool result: %v", second.err)
			case !strings.Contains(strings.ReplaceAll(second.text, ",", ""), "5555"):
				r.outcome, r.detail = fail, r.detail+fmt.Sprintf("args %v, answer “%s”", call.Args, second.text)
			case first.stop != llm.StopToolUse:
				r.outcome = fail
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("args %v → “%s”", call.Args, strings.TrimSpace(second.text))
			}
			return r
		}},
		{"tool_choice named", func(ctx context.Context, p llm.LLMProvider) result {
			c := streamOnce(ctx, p, &llm.ChatRequest{
				Messages:   user("Say hello."),
				Tools:      []llm.ToolDef{addTool},
				ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "add"},
			})
			r := result{usage: c.spent()}
			switch {
			case c.err != nil:
				return failed("%v", c.err)
			case len(c.calls) == 0 || c.calls[0].Name != "add":
				r.outcome, r.detail = fail, fmt.Sprintf("named tool not forced (text “%s”)", c.text)
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("args %v", c.calls[0].Args)
			}
			return r
		}},
		{"final iteration (tool_choice none, tool history)", func(ctx context.Context, p llm.LLMProvider) result {
			// The engine's last iteration: tools declared, a tool call and
			// its result in history, tool_choice none. The backend must
			// accept the request and answer in text.
			call := llm.ToolCall{ID: "call_1", Name: "add", Args: map[string]any{"a": 2, "b": 3}}
			c := streamOnce(ctx, p, &llm.ChatRequest{
				Messages: []llm.Message{
					*llm.UserText("What is 2 + 3? Use the add tool."),
					{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart(call)}},
					{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart(llm.ToolResultData{CallID: call.ID, Name: call.Name, Content: map[string]any{"sum": 5}})}},
					*llm.UserText("[Tool budget exhausted. Give your final text answer now — do not attempt tool calls.]"),
				},
				Tools:      []llm.ToolDef{addTool},
				ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceNone},
			})
			r := result{usage: c.spent()}
			switch {
			case c.err != nil:
				return failed("%v", c.err)
			case len(c.calls) > 0:
				r.outcome, r.detail = fail, fmt.Sprintf("backend ignored tool_choice none: %d call(s), text “%s”", len(c.calls), c.text)
			case !strings.Contains(c.text, "5"):
				r.outcome, r.detail = fail, fmt.Sprintf("text “%s”", c.text)
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("“%s”", strings.TrimSpace(c.text))
			}
			return r
		}},
		{"structured output", func(ctx context.Context, p llm.LLMProvider) result {
			var out struct {
				Answer int `json:"answer"`
			}
			usage, err := p.GenerateStructured(ctx, &llm.ChatRequest{Messages: user("What is 17 + 25?")},
				llm.Schema{
					"type":                 "object",
					"properties":           map[string]any{"answer": map[string]any{"type": "integer"}},
					"required":             []string{"answer"},
					"additionalProperties": false,
				}, &out)
			r := result{usage: usage}
			switch {
			case err != nil:
				return failed("%v", err)
			case out.Answer != 42:
				r.outcome, r.detail = fail, fmt.Sprintf("answer %d", out.Answer)
			case usage.InputTokens == 0:
				r.outcome, r.detail = fail, "no usage reported"
			default:
				r.outcome, r.detail = pass, "answer 42"
			}
			return r
		}},
		{"max_tokens stop", func(ctx context.Context, p llm.LLMProvider) result {
			c := streamOnce(ctx, p, &llm.ChatRequest{
				Messages:  user("Count from 1 to 300, one number per line."),
				MaxTokens: 24,
			})
			r := result{usage: c.spent()}
			switch {
			case c.err != nil:
				return failed("%v", c.err)
			case c.stop != llm.StopMaxTokens:
				r.outcome, r.detail = fail, fmt.Sprintf("stop reason “%s”, want max_tokens (output %d tokens)", c.stop, r.usage.OutputTokens)
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("cut at %d output tokens", r.usage.OutputTokens)
			}
			return r
		}},
		{"reasoning stream", func(ctx context.Context, p llm.LLMProvider) result {
			c := streamOnce(ctx, p, &llm.ChatRequest{Messages: user("Is 221 prime? Think it through, then answer yes or no.")})
			r := result{usage: c.spent(), outcome: info}
			switch {
			case c.err != nil:
				r.outcome, r.detail = fail, c.err.Error()
			case c.thinking:
				r.detail = "thinking deltas observed"
			default:
				r.detail = "no thinking deltas (fine for a non-reasoning model)"
			}
			return r
		}},
		{"engine e2e (tool loop)", func(ctx context.Context, p llm.LLMProvider) result {
			reg := agent.NewToolRegistry()
			_ = reg.Register(calculator())
			engine := agent.NewAgentEngine(agent.AgentConfig{Provider: p, MaxIterations: 4}, reg, agent.NewSessionManager())
			stream, err := engine.StreamChat(ctx, agent.ChatRequest{UserID: "c", SessionID: "c", Message: "Use the calculator tool to multiply 123 by 45, then state the result."})
			if err != nil {
				return failed("%v", err)
			}
			var text strings.Builder
			var r result
			calls := 0
			for chunk := range stream {
				switch chunk.Type {
				case agent.ChunkTypeText:
					text.WriteString(chunk.Content)
				case agent.ChunkTypeToolCall:
					calls++
				case agent.ChunkTypeError:
					if chunk.Usage != nil {
						r.usage = *chunk.Usage
					}
					r.outcome, r.detail = fail, chunk.Error
					return r
				case agent.ChunkTypeDone:
					if chunk.Usage != nil {
						r.usage = *chunk.Usage
					}
				}
			}
			answer := strings.ReplaceAll(text.String(), ",", "")
			switch {
			case calls == 0:
				r.outcome, r.detail = fail, fmt.Sprintf("no tool call; text “%s”", answer)
			case !strings.Contains(answer, "5535"):
				r.outcome, r.detail = fail, fmt.Sprintf("%d call(s); text “%s”", calls, answer)
			default:
				r.outcome, r.detail = pass, fmt.Sprintf("%d call(s) → “%s”", calls, strings.TrimSpace(answer))
			}
			return r
		}},
	}
}

func num(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case string:
		var i int
		_, _ = fmt.Sscan(n, &i)
		return i
	}
	return 0
}

func oneLine(s string, limit int) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > limit {
		return strings.ToValidUTF8(s[:limit], "") + "…"
	}
	return s
}

// calculator is the conformance run's one tool: a multiplication with an
// answer the check can verify.
func calculator() *agent.Tool {
	return &agent.Tool{
		Name:        "calculator",
		ReadOnly:    true,
		Description: "Multiplies two numbers.",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "number", "description": "First factor"},
				"b": map[string]any{"type": "number", "description": "Second factor"},
			},
			"required": []string{"a", "b"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			a, okA := args["a"].(float64)
			b, okB := args["b"].(float64)
			if !okA || !okB {
				return nil, fmt.Errorf("a and b must be numbers")
			}
			return &agent.ToolResult{Data: map[string]any{"result": a * b}}, nil
		},
	}
}
