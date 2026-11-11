package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// maxShellOutput bounds captured stdout+stderr before the model-facing
// truncation applies; output past it is discarded as it arrives.
const maxShellOutput = 128 * 1024

// shellWaitDelay bounds how long output may keep arriving after the
// command exits (a straggler the group kill could not reach) before the
// pipe is cut.
const shellWaitDelay = 2 * time.Second

// shellEnvPassthrough is all a command inherits from the harness process.
// Everything else — API keys loaded from .env above all — stays out: a
// prompt-injected `env` must find nothing worth stealing.
var shellEnvPassthrough = []string{"PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ"}

// shellEnv builds a command's environment: the passthrough list, HOME
// pinned to the session workspace, a dumb terminal.
func shellEnv(workdir string) []string {
	env := []string{"HOME=" + workdir, "TERM=dumb"}
	for _, key := range shellEnvPassthrough {
		if v, ok := os.LookupEnv(key); ok {
			env = append(env, key+"="+v)
		}
	}
	return env
}

// cappedBuffer keeps the first limit bytes written to it and counts the
// rest, so a runaway command cannot grow the harness's memory.
type cappedBuffer struct {
	mu      sync.Mutex
	buf     bytes.Buffer
	limit   int
	dropped int
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	room := b.limit - b.buf.Len()
	if room < len(p) {
		if room > 0 {
			b.buf.Write(p[:room])
		}
		b.dropped += len(p) - max(room, 0)
		return len(p), nil
	}
	return b.buf.Write(p)
}

// BashName is the shell tool's name: the one smaller models already know.
const BashName = "bash"

// ShellConfig parameterizes the bash tool.
type ShellConfig struct {
	WorkspaceRoot string // session workspaces live under this root
	TimeoutS      int    // per-command deadline
}

// NewShell builds the bash tool: bash commands inside the session's
// workspace. Dangerous by design — the config only allows registering it
// behind an explicit opt-in (bootstrap checks Tools.Bash.Enabled). It is
// not a sandbox: a command runs with the harness user's rights. What it
// does guarantee: a scrubbed environment, bounded output, and no process
// outliving its command (the whole process group is killed on return).
func NewShell(cfg ShellConfig) *agent.Tool {
	timeout := time.Duration(cfg.TimeoutS) * time.Second
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	return &agent.Tool{
		Name:        BashName,
		Description: "Runs a bash command inside the session workspace and returns stdout/stderr with the exit code. Working directory is the workspace root of this session. Timeout per command applies.",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{
					"type":        "string",
					"description": "The bash command to run",
				},
			},
			"required": []string{"command"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			command, _ := args["command"].(string)
			if command == "" {
				return nil, fmt.Errorf("bash: command is required")
			}

			workdir, err := SessionWorkspace(cfg.WorkspaceRoot, tc.UserID, tc.SessionID)
			if err != nil {
				return nil, err
			}

			runCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			cmd := exec.CommandContext(runCtx, "bash", "-c", command)
			cmd.Dir = workdir
			cmd.Env = shellEnv(workdir)
			cmd.WaitDelay = shellWaitDelay
			ownProcessGroup(cmd)

			// Output goes through a pipe of our own: Wait then returns when
			// bash exits, not when the last background child closes its
			// copy of stdout — so the group is killed the moment the
			// command is done, before stragglers get to act.
			r, w, err := os.Pipe()
			if err != nil {
				return nil, fmt.Errorf("bash: %w", err)
			}
			cmd.Stdout, cmd.Stderr = w, w
			if err := cmd.Start(); err != nil {
				r.Close()
				w.Close()
				return nil, fmt.Errorf("bash: start: %w", err)
			}
			w.Close() // the command holds its own copy
			out := &cappedBuffer{limit: maxShellOutput}
			copied := make(chan struct{})
			go func() {
				_, _ = io.Copy(out, r)
				close(copied)
			}()

			err = cmd.Wait()
			killProcessGroup(cmd) // stragglers left in the background
			select {
			case <-copied:
			case <-time.After(shellWaitDelay):
				// Something outside the group still holds the pipe.
			}
			r.Close()
			<-copied

			data := map[string]any{
				"command":   command,
				"output":    strings.ToValidUTF8(out.buf.String(), ""),
				"truncated": out.dropped > 0,
			}
			if out.dropped > 0 {
				data["dropped_bytes"] = out.dropped
			}
			if runCtx.Err() == context.DeadlineExceeded {
				data["exit_code"] = -1
				data["error"] = fmt.Sprintf("command timed out after %s", timeout)
				return &agent.ToolResult{Data: data}, nil
			}
			if err != nil {
				if exitErr, ok := err.(*exec.ExitError); ok {
					data["exit_code"] = exitErr.ExitCode()
				} else {
					data["exit_code"] = -1
					data["error"] = err.Error()
				}
			} else {
				data["exit_code"] = 0
			}
			return &agent.ToolResult{Data: data}, nil
		},
	}
}
