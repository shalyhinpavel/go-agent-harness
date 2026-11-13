package safename

import (
	"strings"
	"testing"
)

func TestOf_PlainIDsAreUntouched(t *testing.T) {
	for _, id := range []string{"s1", "3f2b8c1e-aaaa-bbbb-cccc-0123456789ab", "team_1", strings.Repeat("a", 100)} {
		if got := Of(id, "x"); got != id {
			t.Errorf("Of(%q) = %q, want it unchanged", id, got)
		}
	}
	if Of("", "fallback") != "fallback" {
		t.Error("an empty id takes the fallback")
	}
}

func TestOf_DifferentIDsNeverShareAName(t *testing.T) {
	ids := []string{
		"team/1", "team:1", "team.1", "team-1", "team 1", "../team-1", "TEAM-1", "Team-1", "A", "a",
		"αβγδεζ", "你好", "a@b.c", "a-b.c", strings.Repeat("a", 101), strings.Repeat("a", 102),
	}
	seen := map[string]string{}
	for _, id := range ids {
		name := Of(id, "x")
		// Compared case-folded: the filesystem may do the same.
		if prev, dup := seen[strings.ToLower(name)]; dup {
			t.Errorf("%q and %q both map to %q", prev, id, name)
		}
		seen[strings.ToLower(name)] = id
		if strings.ContainsAny(name, "/\\:. ") || len(name) > 100 {
			t.Errorf("Of(%q) = %q is not a safe component", id, name)
		}
	}
}
