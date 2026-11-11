package main

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"
)

// A request already running when the server's context ends must see its own
// context end too, so a signal stops the turn instead of leaving Shutdown to
// wait out its timeout.
func TestNewHTTPServer_ServerContextEndsRunningRequests(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	inHandler := make(chan struct{})
	unblocked := make(chan struct{})
	srv := newHTTPServer(ctx, "127.0.0.1:0", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(inHandler)
		<-r.Context().Done()
		close(unblocked)
	}))

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	go func() {
		if resp, err := http.Get("http://" + ln.Addr().String() + "/"); err == nil {
			resp.Body.Close()
		}
	}()
	select {
	case <-inHandler:
	case <-time.After(5 * time.Second):
		t.Fatal("the request never reached the handler")
	}

	cancel()
	select {
	case <-unblocked:
	case <-time.After(2 * time.Second):
		t.Fatal("cancelling the server's context must end a running request")
	}
}
