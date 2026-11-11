package llm

import (
	"encoding/json"
	"strings"
)

// ExtractJSON pulls the first JSON object out of a model reply: tolerates
// markdown fences and surrounding prose, which is what prompt-guided
// models produce no matter how sternly they are asked not to. Returns ""
// when the reply holds no object at all.
func ExtractJSON(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// Strip a fenced block wholesale when present.
	if strings.HasPrefix(s, "```") {
		if _, rest, ok := strings.Cut(s, "\n"); ok {
			s = rest
		}
		s = strings.TrimPrefix(strings.TrimSpace(s), "```json")
		s = strings.TrimSuffix(strings.TrimSpace(s), "```")
		s = strings.TrimSpace(s)
	}
	// Each brace that opens a balanced span is a candidate. Prose ahead of the
	// object can hold balanced braces of its own ("{user}: {...}"), so the
	// first span counts only when it is valid JSON.
	for start := strings.IndexByte(s, '{'); start >= 0; {
		end := closingBrace(s, start)
		if end < 0 {
			return "" // never closed: a cut-off reply holds no usable object
		}
		if candidate := s[start:end]; json.Valid([]byte(candidate)) {
			return candidate
		}
		next := strings.IndexByte(s[start+1:], '{')
		if next < 0 {
			break
		}
		start += 1 + next
	}
	return ""
}

// closingBrace returns the index just past the brace that closes the object
// opened at s[start], or -1 when it never closes. Prose after the object is
// not walked, so its braces cannot swallow the object; braces inside strings
// do not count.
func closingBrace(s string, start int) int {
	depth, inString, escaped := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case escaped:
			escaped = false
		case inString:
			switch c {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
		case c == '"':
			inString = true
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// SchemaPromptNote is the instruction appended to a prompt when a backend
// cannot constrain output natively: the schema rides in the prompt and
// the reply goes through ExtractJSON.
func SchemaPromptNote(schemaJSON string) string {
	return "\n\nReply with ONLY one valid JSON object matching this JSON Schema. No prose, no markdown fences:\n" + schemaJSON
}

// WithSchemaNote returns a copy of req whose final user message carries
// the schema note (a new user message when the last one is not the
// user's). Text parts only: the prompt-guided path is the compatibility
// fallback, not the place for multimodal input. The caller's request is
// left untouched.
func WithSchemaNote(req *ChatRequest, schemaJSON string) *ChatRequest {
	note := SchemaPromptNote(schemaJSON)
	out := *req
	msgs := make([]Message, len(req.Messages))
	copy(msgs, req.Messages)
	if n := len(msgs); n > 0 && msgs[n-1].Role == RoleUser {
		var sb strings.Builder
		for _, part := range msgs[n-1].Parts {
			if part.Kind == PartText {
				sb.WriteString(part.Text)
			}
		}
		msgs[n-1] = *UserText(sb.String() + note)
	} else {
		msgs = append(msgs, *UserText(strings.TrimSpace(note)))
	}
	out.Messages = msgs
	return &out
}
