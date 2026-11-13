package agent

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestCutUTF8_NeverSplitsACharacter(t *testing.T) {
	s := strings.Repeat("αβγ", 50) // two bytes per letter
	for n := 1; n < len(s); n++ {
		got := cutUTF8(s, n)
		if !utf8.ValidString(got) || len(got) > n {
			t.Fatalf("cutUTF8(.., %d) = %d bytes, valid=%v", n, len(got), utf8.ValidString(got))
		}
	}
	if cutUTF8("short", 100) != "short" {
		t.Error("a string under the limit stays whole")
	}
}
