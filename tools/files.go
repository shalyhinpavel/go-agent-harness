package tools

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/shalyhinpavel/go-agent-harness/agent"
	"github.com/shalyhinpavel/go-agent-harness/internal/safename"
	"github.com/shalyhinpavel/go-agent-harness/llm"
)

// maxReadBytes caps raw file reads before parsing; oversized content is
// the model-facing truncation's job, this is just a sanity stop.
const maxReadBytes = 8 * 1024 * 1024

// workspaceDir resolves the session's own directory under root.
func workspaceDir(root string, tc *agent.ToolContext) (string, error) {
	return SessionWorkspace(root, tc.UserID, tc.SessionID)
}

// Names of the file tools: one operation each, no `op` switch — the shape
// smaller models call more reliably than a polymorphic tool.
const (
	ReadFileName  = "read_file"
	WriteFileName = "write_file"
	ListFilesName = "list_files"
)

// IsFileTool reports whether name is a workspace file tool: the ones that
// need a workspace root.
func IsFileTool(name string) bool {
	switch name {
	case ReadFileName, WriteFileName, ListFilesName:
		return true
	}
	return false
}

// HasFileTool reports whether any of the names is a workspace file tool.
func HasFileTool(names []string) bool {
	for _, n := range names {
		if IsFileTool(n) {
			return true
		}
	}
	return false
}

// NewReadFile builds read_file over the per-session workspace under root.
func NewReadFile(root string) *agent.Tool {
	return &agent.Tool{
		Name:        ReadFileName,
		Description: "Read a file from the session workspace. Known formats (xlsx, csv, json, pdf, docx, md, txt) are parsed into model-friendly text. The path is relative to the workspace.",
		ReadOnly:    true,
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string", "description": "Relative file path"},
			},
			"required": []string{"path"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			dir, err := workspaceDir(root, tc)
			if err != nil {
				return nil, err
			}
			path, _ := args["path"].(string)
			return readWorkspaceFile(dir, path)
		},
	}
}

// NewWriteFile builds write_file over the per-session workspace under root.
func NewWriteFile(root string) *agent.Tool {
	return &agent.Tool{
		Name:        WriteFileName,
		Description: "Write a text file into the session workspace (creates or overwrites). The path is relative to the workspace.",
		Parameters: llm.Schema{
			"type": "object",
			"properties": map[string]any{
				"path":    map[string]any{"type": "string", "description": "Relative file path"},
				"content": map[string]any{"type": "string", "description": "File content"},
			},
			"required": []string{"path", "content"},
		},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			dir, err := workspaceDir(root, tc)
			if err != nil {
				return nil, err
			}
			path, _ := args["path"].(string)
			content, _ := args["content"].(string)
			return writeWorkspaceFile(dir, path, content)
		},
	}
}

// NewListFiles builds list_files over the per-session workspace under root.
func NewListFiles(root string) *agent.Tool {
	return &agent.Tool{
		Name:        ListFilesName,
		Description: "List the files in the session workspace.",
		ReadOnly:    true,
		Parameters:  llm.Schema{"type": "object", "properties": map[string]any{}},
		Execute: func(ctx context.Context, tc *agent.ToolContext, args map[string]any) (*agent.ToolResult, error) {
			dir, err := workspaceDir(root, tc)
			if err != nil {
				return nil, err
			}
			return listWorkspace(dir)
		},
	}
}

// SessionWorkspace returns (creating it) the session's directory under
// root, scoped to userID so an evicted or restarted in-memory session id
// never shares files across users. Both ids are sanitized so nothing
// escapes the root.
func SessionWorkspace(root, userID, sessionID string) (string, error) {
	if root == "" {
		return "", fmt.Errorf("files: workspace root is not configured")
	}
	dir := filepath.Join(root, safename.Of(userID, "anonymous"), SafeName(sessionID))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("files: create workspace: %w", err)
	}
	return dir, nil
}

// cleanRel turns a model-supplied path into a clean path relative to the
// workspace. Anchoring it at "/" first removes any "..", so a path can only
// name something inside; symlinks are the job of openWorkspace.
func cleanRel(rel string) (string, error) {
	clean := strings.TrimPrefix(filepath.Clean("/"+strings.ReplaceAll(rel, "\\", "/")), "/")
	if clean == "" {
		return "", fmt.Errorf("files: path is required")
	}
	return filepath.FromSlash(clean), nil
}

// openWorkspace opens dir as a root: nothing done through it can leave dir,
// whatever the symlinks inside it point at (a bash command or an MCP server
// can create them).
func openWorkspace(dir string) (*os.Root, error) {
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, fmt.Errorf("files: open workspace: %w", err)
	}
	return root, nil
}

// maxListed bounds one list_files answer.
const maxListed = 500

// listWorkspace lists every file under the workspace, subdirectories
// included, as paths relative to it.
func listWorkspace(dir string) (*agent.ToolResult, error) {
	root, err := openWorkspace(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()

	files := []map[string]any{}
	truncated := false
	err = fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || path == "." {
			return nil // an unreadable entry is skipped, not fatal
		}
		if len(files) >= maxListed {
			truncated = true
			return fs.SkipAll
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		files = append(files, map[string]any{"name": path, "size": info.Size(), "dir": d.IsDir()})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("files: list: %w", err)
	}
	data := map[string]any{"files": files, "count": len(files)}
	if truncated {
		data["note"] = fmt.Sprintf("only the first %d entries are listed", maxListed)
	}
	return &agent.ToolResult{Data: data}, nil
}

func readWorkspaceFile(dir, rel string) (*agent.ToolResult, error) {
	clean, err := cleanRel(rel)
	if err != nil {
		return nil, err
	}
	root, err := openWorkspace(dir)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	f, err := root.Open(clean)
	if err != nil {
		return nil, fmt.Errorf("files: read: %w", err)
	}
	defer f.Close()
	// Read one byte past the cap: a bigger file is never loaded whole.
	raw, err := io.ReadAll(io.LimitReader(f, maxReadBytes+1))
	if err != nil {
		return nil, fmt.Errorf("files: read: %w", err)
	}
	ext := filepath.Ext(strings.ToLower(rel))
	truncated := len(raw) > maxReadBytes
	if truncated {
		// A cut zip or PDF is not a smaller document, only a broken one.
		if ext == ".pdf" || ext == ".docx" || ext == ".xlsx" {
			return nil, fmt.Errorf("files: %s is larger than %d MB, too big to parse", rel, maxReadBytes>>20)
		}
		raw = raw[:maxReadBytes]
	}

	content, note, err := ParseDocument(ext, raw)
	if err != nil {
		return nil, fmt.Errorf("files: parse %s: %w", rel, err)
	}
	if truncated {
		note = strings.TrimSpace(note + fmt.Sprintf(" only the first %d MB were read", maxReadBytes>>20))
	}
	data := map[string]any{"path": rel, "content": content}
	if note != "" {
		data["note"] = note
	}
	return &agent.ToolResult{Data: data}, nil
}

func writeWorkspaceFile(dir, rel, content string) (*agent.ToolResult, error) {
	clean, err := cleanRel(rel)
	if err != nil {
		return nil, err
	}
	if err := StoreFile(dir, clean, []byte(content)); err != nil {
		return nil, err
	}
	return &agent.ToolResult{Data: map[string]any{
		"path": rel, "bytes": len(content),
	}}, nil
}

// SafeName keeps a generated path component (a session id) filesystem-safe.
// Everything that lays out workspace paths uses it, so the file tools and
// the HTTP server agree on directory names.
func SafeName(name string) string {
	return safename.Of(name, "session")
}

// base64OrEmpty is a helper for binary payloads.
func base64OrEmpty(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return base64.StdEncoding.EncodeToString(b)
}

// StoreFile writes data to the workspace directory dir under the relative
// path rel, creating parent directories. It cannot write outside dir, even
// through a symlink planted inside it.
func StoreFile(dir, rel string, data []byte) error {
	root, err := openWorkspace(dir)
	if err != nil {
		return err
	}
	defer root.Close()
	if parent := filepath.Dir(rel); parent != "." {
		if err := root.MkdirAll(parent, 0o755); err != nil {
			return fmt.Errorf("files: write: %w", err)
		}
	}
	if err := root.WriteFile(rel, data, 0o644); err != nil {
		return fmt.Errorf("files: write: %w", err)
	}
	return nil
}
