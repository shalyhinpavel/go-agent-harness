// Package agent is the engine: it runs the tool-calling loop over any
// llm.LLMProvider, keeps conversations in sessions and streams every step
// of a turn as typed chunks.
//
// An engine is built from three parts:
//
//	registry := agent.NewToolRegistry()          // the tools the model may call
//	sessions := agent.NewSessionManager()        // conversations, with TTL and LRU
//	engine := agent.NewAgentEngine(agent.AgentConfig{
//		Provider:     provider,
//		SystemPrompt: "You are a helpful assistant.",
//	}, registry, sessions)
//
// StreamChat runs one turn and returns a channel of StreamChunk values:
// text, thinking, tool calls, optional approval requests, tool results, and
// finally one done or error chunk carrying the turn's token usage. Turns of
// the same session run one at a time.
//
// Policy and human-in-the-loop controls plug in through AgentConfig.Hooks
// and AgentConfig.Approver. Memory and web search stay outside the engine:
// attach them as tools, MCP servers, or through AgentConfig.PromptAugmenter.
package agent
