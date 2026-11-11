// Package safename turns identifiers (session ids, user ids) into names that
// are safe as one path component. The harness lays out workspaces, traces and
// memory by identifier, so two identifiers must never share a name: that
// would give two sessions, or two users, one directory.
package safename

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// maxPlain is the longest identifier kept as it is; the longest name Of
// returns is shorter than 100 bytes, far below any filesystem limit.
const maxPlain = 100

// Of returns a path-safe name for id. An identifier made only of lowercase
// letters, digits, '-' and '_' (a UUID, a slug) is returned unchanged.
// Anything else has its unsafe characters replaced, is lowercased and gets a
// short hash of the original appended, so "team/1" and "team:1" get
// different names, and so do "Team" and "team" — which one case-insensitive
// filesystem (the macOS and Windows default) would otherwise merge. An
// empty id returns fallback.
func Of(id, fallback string) string {
	if id == "" {
		return fallback
	}
	if len(id) <= maxPlain && strings.IndexFunc(id, unsafe) < 0 {
		return id
	}
	plain := strings.Map(func(r rune) rune {
		if unsafe(r) {
			return '-'
		}
		return r
	}, strings.ToLower(id))
	if len(plain) > 80 {
		plain = plain[:80]
	}
	sum := sha256.Sum256([]byte(id))
	return plain + "-" + hex.EncodeToString(sum[:6])
}

func unsafe(r rune) bool {
	switch {
	case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-', r == '_':
		return false
	}
	return true
}
