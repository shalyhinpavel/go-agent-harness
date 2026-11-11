package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/internal/bootstrap"
	"github.com/shalyhinpavel/go-agent-harness/internal/config"
	"github.com/shalyhinpavel/go-agent-harness/server"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

// runServe serves the HTTP + SSE API until SIGINT or SIGTERM.
func runServe(args []string) int {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	configPath := configFlag(fs)
	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "usage: harness serve [flags]\n\nServes GET /health, POST /v1/chat/stream (SSE) and POST /v1/chat/approve.\nThere is no authentication: keep it on loopback or put an authenticating proxy in front.")
		fs.PrintDefaults()
	}
	_ = fs.Parse(args)

	cfg, err := config.Load(*configPath)
	if err != nil {
		return fail1(err)
	}
	approvalTimeout, err := cfg.Server.ParsedApprovalTimeout()
	if err != nil {
		return fail1(err)
	}
	h, err := bootstrap.BuildEngine(cfg)
	if err != nil {
		return fail1(err)
	}
	defer h.Close()

	opts := server.Options{
		CORSOrigins:     cfg.Server.CORSOrigins,
		MaxBodyMB:       cfg.Server.MaxBodyMB,
		RequireApproval: cfg.Server.RequireApproval,
		ApprovalTimeout: approvalTimeout,
	}
	for _, t := range h.Registry.List() {
		if tools.IsFileTool(t.Name) {
			opts.FilesDir = cfg.Tools.Files.Dir
			break
		}
	}
	addr := net.JoinHostPort(cfg.Server.Host, strconv.Itoa(cfg.Server.Port))
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	srv := newHTTPServer(ctx, addr, server.NewHandler(h.Engine, opts))
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	slog.Info("serving (no authentication: keep it on loopback or behind an authenticating proxy)", "addr", addr)

	select {
	case err := <-errc:
		return fail1(err)
	case <-ctx.Done():
	}
	slog.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fail1(err)
	}
	return 0
}

// newHTTPServer roots every request context at ctx. Shutdown waits for live
// requests but does not cancel them, so without this a running SSE turn would
// hold the server until its timeout; with it, the signal cancels the turn and
// the request ends.
func newHTTPServer(ctx context.Context, addr string, handler http.Handler) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		BaseContext:       func(net.Listener) context.Context { return ctx },
	}
}
