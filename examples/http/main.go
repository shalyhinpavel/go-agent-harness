// Http serves the agent over HTTP with your own authentication in front.
// server.NewHandler is a plain http.Handler: wrap it, mount it under a
// prefix, or put it in any router.
//
//	API_TOKEN=secret go run ./examples/http
//	curl -N -H 'Authorization: Bearer secret' localhost:8080/v1/chat/stream \
//	     -d '{"message":"hello"}'
package main

import (
	"crypto/subtle"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/examples/internal/demo"
	"github.com/shalyhinpavel/go-agent-harness/server"
	"github.com/shalyhinpavel/go-agent-harness/tools"
)

// bearer rejects requests without the expected token. /health stays open, and
// so do CORS preflights: browsers send them without credentials.
func bearer(token string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if r.URL.Path != "/health" && r.Method != http.MethodOptions && subtle.ConstantTimeCompare([]byte(got), []byte(token)) != 1 {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	token := os.Getenv("API_TOKEN")
	if token == "" {
		log.Fatal("set API_TOKEN: the handler has no authentication of its own")
	}
	provider, err := demo.Provider()
	if err != nil {
		log.Fatal(err)
	}

	registry := agent.NewToolRegistry()
	if err := registry.Register(tools.NewWebFetch(tools.WebFetchConfig{BlockPrivate: true})); err != nil {
		log.Fatal(err)
	}
	engine := agent.NewAgentEngine(agent.AgentConfig{
		Provider:      provider,
		SystemPrompt:  "You are a helpful assistant.",
		MaxIterations: 8,
	}, registry, agent.NewSessionManager(agent.WithSessionTTL(time.Hour)))

	srv := &http.Server{
		Addr: "127.0.0.1:8080",
		Handler: bearer(token, server.NewHandler(engine, server.Options{
			// The user id comes from the authenticated caller, never from
			// the request body: otherwise any client could read another
			// user's session by naming them.
			Identify: func(*http.Request) (server.Identity, error) {
				return server.Identity{UserID: "api-client"}, nil
			},
		})),
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf("listening on %s", srv.Addr)
	log.Fatal(srv.ListenAndServe())
}
