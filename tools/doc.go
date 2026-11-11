// Package tools holds the built-in agent tools:
//
//   - web_fetch (NewWebFetch): a page as readable text with its links kept,
//     served in windows the model pages through; optional anti-SSRF guard;
//   - read_file, write_file, list_files (NewReadFile, NewWriteFile,
//     NewListFiles): a per-session workspace, with PDF, DOCX, XLSX, CSV and
//     JSON parsed on read;
//   - bash (NewShell): a shell in the session workspace. It scrubs the
//     environment and bounds output and time, but it is not a sandbox.
//
// Each constructor returns an *agent.Tool to register directly.
// RegisterSelected builds a set of them by name, which is how the config
// file selects tools. There is deliberately no search tool: bring the
// search API you use as your own tool or through an MCP server.
package tools
