package llm

import "testing"

func TestExtractJSON(t *testing.T) {
	cases := map[string]string{
		`{"a":1}`:                               `{"a":1}`,
		"```json\n{\"a\":1}\n```":               `{"a":1}`,
		"Here you go:\n```json\n{\"a\":1}\n```": `{"a":1}`,
		`Sure! {"a":1} hope that helps`:         `{"a":1}`,
		"no json here":                          "",
		"":                                      "",
		// Prose after the object may hold braces of its own.
		`{"a":1} (see {note})`:           `{"a":1}`,
		`{"a":{"b":[1,2]},"c":"x"} tail`: `{"a":{"b":[1,2]},"c":"x"}`,
		// Braces and escaped quotes inside strings do not count.
		`{"a":"} \" {"} after {x}`: `{"a":"} \" {"}`,
		// A cut-off object is not an object.
		`{"a":{"b":1}`: "",
		// Prose ahead of the object can hold balanced braces of its own: a
		// span counts only when it is valid JSON.
		`{user}: {"answer": 42}`:   `{"answer": 42}`,
		`{x} and {y} then {"a":1}`: `{"a":1}`,
		`{user}: {"answer": 42`:    "",
	}
	for in, want := range cases {
		if got := ExtractJSON(in); got != want {
			t.Errorf("ExtractJSON(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWithSchemaNote_LeavesCallerRequestUntouched(t *testing.T) {
	req := &ChatRequest{Messages: []Message{*UserText("pick one")}}
	got := WithSchemaNote(req, `{"type":"object"}`)
	if req.Messages[0].Parts[0].Text != "pick one" {
		t.Fatalf("caller request mutated: %q", req.Messages[0].Parts[0].Text)
	}
	if len(got.Messages) != 1 || got.Messages[0].Parts[0].Text == "pick one" {
		t.Fatalf("schema note not appended to the last user message: %+v", got.Messages)
	}

	tail := &ChatRequest{Messages: []Message{*AssistantText("previous")}}
	if got := WithSchemaNote(tail, "{}"); len(got.Messages) != 2 || got.Messages[1].Role != RoleUser {
		t.Fatalf("non-user tail must get a new user message: %+v", got.Messages)
	}
}
