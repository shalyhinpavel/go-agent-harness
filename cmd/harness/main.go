// Command harness runs the agent from the command line: a chat in the
// terminal, the HTTP API, or a conformance check of a model backend.
//
//	harness                    chat in the terminal
//	harness chat "question"    answer one question and exit
//	harness serve              HTTP + SSE API on 127.0.0.1:8080
//	harness check              check a backend against the provider contract
//
// The model provider is picked from the API key in the environment or in a
// .env file (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY);
// configs/config.yaml, when present, sets everything else.
package main

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	"github.com/joho/godotenv"
)

// logLevel is the verbosity of the diagnostics on stderr; the chat commands
// lower it to warnings so they do not interleave with the conversation.
var logLevel = new(slog.LevelVar)

const usage = `harness — a provider-agnostic agent from the command line

usage:
  harness [chat] [flags] [message]   chat in the terminal; with a message, answer it and exit
  harness serve [flags]              HTTP + SSE API (no authentication: keep it on loopback)
  harness check [flags]              check a model backend against the provider contract

The provider is picked from ANTHROPIC_API_KEY, OPENAI_API_KEY (+ OPENAI_BASE_URL
for a local server) or OPENROUTER_API_KEY, read from the environment or .env.
Run "harness <command> -h" for the command's flags.
`

func main() {
	_ = godotenv.Load()
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel})))

	cmd, args := "chat", os.Args[1:]
	if len(args) > 0 {
		switch args[0] {
		case "chat", "serve", "check":
			cmd, args = args[0], args[1:]
		case "help", "-h", "-help", "--help":
			fmt.Print(usage)
			return
		}
	}

	switch cmd {
	case "serve":
		os.Exit(runServe(args))
	case "check":
		os.Exit(runCheck(args))
	default:
		os.Exit(runChat(args))
	}
}

// configFlag registers the -config flag every command shares.
func configFlag(fs *flag.FlagSet) *string {
	return fs.String("config", "configs/config.yaml", "YAML config (a missing file means defaults)")
}

// fail1 reports a startup error and returns the exit status for it.
func fail1(err error) int {
	fmt.Fprintln(os.Stderr, "harness:", err)
	return 1
}
