// Package prompts serves prompt templates from a directory on disk with
// embedded defaults as fallback — edit a file, no rebuild needed.
package prompts

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

//go:embed defaults/*.md
var embedded embed.FS

// Loader resolves prompt files: the configured directory first, embedded
// defaults second.
type Loader struct {
	Dir string // may be empty: embedded defaults only
}

// New creates a loader reading overrides from dir.
func New(dir string) *Loader {
	return &Loader{Dir: dir}
}

// Load returns the prompt text for a logical name like "system".
// File extension is optional and normalized to .md.
func (l *Loader) Load(name string) (string, error) {
	name = strings.TrimSuffix(name, filepath.Ext(name)) + ".md"

	if l.Dir != "" {
		path := filepath.Join(l.Dir, name)
		if data, err := os.ReadFile(path); err == nil {
			return string(data), nil
		} else if !os.IsNotExist(err) {
			return "", fmt.Errorf("prompts: read %s: %w", path, err)
		}
	}

	data, err := embedded.ReadFile("defaults/" + name)
	if err != nil {
		return "", fmt.Errorf("prompts: %q not found on disk or in defaults", name)
	}
	return string(data), nil
}
