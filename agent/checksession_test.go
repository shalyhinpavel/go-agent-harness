package agent_test

import (
	"context"
	"errors"
	"testing"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// After a restart nothing is in memory, so the owner has to come from the
// store; otherwise another user could open the session the first one made.
func TestCheckSession_OwnerSurvivesARestart(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	store, err := agent.NewFileSessionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	before := agent.NewSessionManager(agent.WithSessionStore(store))
	if _, err := before.GetOrCreateSession(ctx, "s1", "alice"); err != nil {
		t.Fatal(err)
	}

	reopened, err := agent.NewFileSessionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	after := agent.NewSessionManager(agent.WithSessionStore(reopened))
	engine := agent.NewAgentEngine(agent.AgentConfig{Provider: llm.NewMockProvider()}, agent.NewToolRegistry(), after)

	if err := engine.CheckSession(ctx, "mallory", "s1"); !errors.Is(err, agent.ErrSessionOwner) {
		t.Fatalf("another user on a stored session: err = %v, want ErrSessionOwner", err)
	}
	if err := engine.CheckSession(ctx, "alice", "s1"); err != nil {
		t.Fatalf("the owner after a restart: err = %v", err)
	}
	if err := engine.CheckSession(ctx, "mallory", "fresh"); err != nil {
		t.Fatalf("a session that does not exist yet must pass: err = %v", err)
	}
}
