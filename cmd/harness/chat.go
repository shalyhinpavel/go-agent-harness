package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/bootstrap"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

// runChat chats with the agent in the terminal, or answers the message
// given on the command line and exits.
func runChat(args []string) int {
	logLevel.Set(slog.LevelWarn)
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	configPath := configFlag(fs)
	model := fs.String("model", "", "model id (overrides the config)")
	thinking := fs.Bool("thinking", false, "show the model's reasoning as it streams")
	approve := fs.Bool("approve", false, "ask before running non-read-only tools")
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: harness [chat] [flags] [message]\n\nWithout a message: an interactive chat. With one: answer it and exit.")
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
	cfg.Provider = bootstrap.Resolve(cfg.Provider)
	h, err := bootstrap.BuildEngine(cfg)
	if err != nil {
		return fail1(err)
	}
	defer h.Close()

	out := newPrinter(os.Stdout, *thinking)
	out.approve = *approve
	if *approve || fs.NArg() == 0 {
		out.in = readLines(os.Stdin)
	}
	session := uuid.NewString()

	if fs.NArg() > 0 {
		if err := out.turn(h.Engine, session, strings.Join(fs.Args(), " ")); err != nil {
			return fail1(err)
		}
		return 0
	}

	out.banner(cfg, h.Registry)
	for {
		out.prompt()
		raw, ok := <-out.in
		if !ok {
			fmt.Fprintln(out.w)
			return 0
		}
		line := strings.TrimSpace(raw)
		switch line {
		case "":
			continue
		case "/exit", "/quit", "exit", "quit":
			return 0
		case "/new":
			session = uuid.NewString()
			out.note("new conversation")
			continue
		}
		// A failed turn is shown and the chat goes on: the next message
		// may well succeed (a timeout, a rate limit).
		if err := out.turn(h.Engine, session, line); err != nil {
			out.note("error: " + err.Error())
		}
	}
}

// readLines scans lines on a single goroutine so a Ctrl+C during an approval
// prompt can abandon waiting without racing the main REPL loop on the scanner.
func readLines(r io.Reader) <-chan string {
	ch := make(chan string)
	go func() {
		defer close(ch)
		sc := bufio.NewScanner(r)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		for sc.Scan() {
			ch <- sc.Text()
		}
	}()
	return ch
}

// printer renders a turn for a terminal: text as it streams, tool calls
// and their outcomes on lines of their own, and the turn's bill.
type printer struct {
	w        io.Writer
	in       <-chan string
	approve  bool
	color    bool
	thinking bool
	midLine  bool // the last write did not end with a newline
}

func newPrinter(f *os.File, thinking bool) *printer {
	color := false
	if st, err := f.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 && os.Getenv("NO_COLOR") == "" {
		color = true
	}
	return &printer{w: f, color: color, thinking: thinking}
}

func (p *printer) style(code, s string) string {
	if !p.color {
		return s
	}
	return "\x1b[" + code + "m" + s + "\x1b[0m"
}

func (p *printer) write(s string) {
	if s == "" {
		return
	}
	fmt.Fprint(p.w, s)
	p.midLine = !strings.HasSuffix(s, "\n")
}

// line writes s on a line of its own.
func (p *printer) line(s string) {
	if p.midLine {
		p.write("\n")
	}
	p.write(s + "\n")
}

func (p *printer) note(s string) { p.line(p.style("2", s)) }

func (p *printer) prompt() {
	p.line("")
	p.write(p.style("1;36", "> "))
	p.midLine = false // the user's Enter ends the line
}

func (p *printer) banner(cfg config.Config, reg *agent.ToolRegistry) {
	pr := cfg.Provider
	head := "go-agent-harness · " + pr.Kind
	if pr.Model != "" {
		head += " · " + pr.Model
	}
	p.line(p.style("1", head))
	if pr.Kind == "mock" {
		p.line(p.style("33", "No API key found: replies come from a built-in mock. Set ANTHROPIC_API_KEY,\n"+
			"OPENAI_API_KEY or OPENROUTER_API_KEY (environment or .env) to chat with a real model."))
	}
	var names []string
	files := false
	for _, t := range reg.List() {
		names = append(names, t.Name)
		files = files || tools.IsFileTool(t.Name)
	}
	sort.Strings(names)
	toolLine := "tools: none"
	if len(names) > 0 {
		toolLine = "tools: " + strings.Join(names, ", ")
	}
	if files {
		toolLine += " (files in " + cfg.Tools.Files.Dir + ")"
	}
	p.note(toolLine)
	p.note("/new starts a new conversation, /exit quits, Ctrl+C stops a reply")
}

// turn runs one message through the engine and prints it. Ctrl+C during
// the turn cancels the turn, not the program.
func (p *printer) turn(engine *agent.AgentEngine, session, message string) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	req := agent.ChatRequest{UserID: "cli", SessionID: session, Message: message}
	var pendingMu sync.Mutex
	pending := map[string]chan agent.ApprovalDecision{}
	slot := func(id string) chan agent.ApprovalDecision {
		pendingMu.Lock()
		defer pendingMu.Unlock()
		ch, ok := pending[id]
		if !ok {
			ch = make(chan agent.ApprovalDecision, 1)
			pending[id] = ch
		}
		return ch
	}
	if p.approve {
		req.Approver = func(ctx context.Context, ar agent.ApprovalRequest) (agent.ApprovalDecision, error) {
			select {
			case dec := <-slot(ar.ToolCallID):
				return dec, nil
			case <-ctx.Done():
				return agent.ApprovalDecision{}, ctx.Err()
			}
		}
	}

	start := time.Now()
	stream, err := engine.StreamChat(ctx, req)
	if err != nil {
		return err
	}
	var turnErr error
	inThinking := false
	for c := range stream {
		if inThinking && c.Type != agent.ChunkTypeThinking {
			p.line("")
			inThinking = false
		}
		switch c.Type {
		case agent.ChunkTypeThinking:
			if p.thinking {
				inThinking = true
				p.write(p.style("2;3", c.Content))
			}
		case agent.ChunkTypeText:
			p.write(c.Content)
		case agent.ChunkTypeToolCall:
			args, _ := json.Marshal(c.ToolArgs)
			p.line(p.style("2", "→ "+c.ToolName+" "+oneLine(string(args), 100)))
		case agent.ChunkTypeApprovalRequired:
			if !p.approve {
				continue
			}
			if p.midLine {
				p.write("\n")
			}
			p.write(p.style("33", fmt.Sprintf("? allow %s? [y/N]: ", c.ToolName)))
			p.midLine = false
			select {
			case raw, ok := <-p.in:
				if !ok {
					p.line("")
					slot(c.ToolCallID) <- agent.ApprovalDecision{Approved: false, Reason: "input closed"}
					continue
				}
				trimmed := strings.TrimSpace(raw)
				lower := strings.ToLower(trimmed)
				if lower == "y" || lower == "yes" {
					slot(c.ToolCallID) <- agent.ApprovalDecision{Approved: true}
				} else {
					reason := ""
					if lower != "" && lower != "n" && lower != "no" {
						reason = trimmed
					}
					slot(c.ToolCallID) <- agent.ApprovalDecision{Approved: false, Reason: reason}
				}
			case <-ctx.Done():
				p.line("")
			}
		case agent.ChunkTypeToolResult:
			if r := c.ToolResult; r != nil && r.Error != "" {
				p.line(p.style("2;31", "✗ "+c.ToolName+": "+oneLine(r.Error, 100)))
			} else {
				p.line(p.style("2", "✓ "+c.ToolName))
			}
		case agent.ChunkTypeError:
			turnErr = errors.New(c.Error)
			if ctx.Err() != nil {
				turnErr = errors.New("interrupted")
			}
		case agent.ChunkTypeDone:
			if u := c.Usage; u != nil && u.InputTokens+u.OutputTokens > 0 {
				p.note(fmt.Sprintf("(%s in · %s out · %.1fs)", tokens(u.InputTokens), tokens(u.OutputTokens), time.Since(start).Seconds()))
			}
		}
	}
	if p.midLine {
		p.write("\n")
	}
	return turnErr
}

// tokens shortens a token count for the bill: 950, 12.3k.
func tokens(n int) string {
	if n < 1000 {
		return fmt.Sprint(n)
	}
	return fmt.Sprintf("%.1fk", float64(n)/1000)
}
