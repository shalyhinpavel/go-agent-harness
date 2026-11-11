// Package memory connects long-term user memory to the agent without
// putting a memory engine into it. There are two ports:
//
//   - injection: BuildAugmentedSystemPrompt layers a user dossier and
//     retrieved facts onto the system prompt, wired in through
//     agent.AgentConfig.PromptAugmenter;
//   - a tool: NewSearchTool lets the model query a MemoryStore itself.
//
// MemoryStore is the storage port. Two stores ship as starting points:
// InMemoryStore for tests and local runs, and FileStore, which keeps each
// fact as a human-editable markdown note with lexical search. Vector search
// stays a port: pass an embedding function to NewSearchTool and implement
// SearchFacts over the vector store of your choice.
package memory
