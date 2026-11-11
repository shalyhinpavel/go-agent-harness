package memory

import (
	"strings"
	"testing"
)

func TestBuildAugmentedSystemPrompt_BaseOnly(t *testing.T) {
	basePrompt := "You are a helpful assistant."

	result := BuildAugmentedSystemPrompt(basePrompt, nil, nil)

	if !strings.Contains(result, basePrompt) {
		t.Error("Expected base prompt to be included")
	}
	if strings.Contains(result, "USER DOSSIER") {
		t.Error("Should not include dossier section when nil")
	}
	if strings.Contains(result, "RELEVANT USER FACTS") {
		t.Error("Should not include facts section when nil")
	}
}

func TestBuildAugmentedSystemPrompt_WithDossier(t *testing.T) {
	basePrompt := "You are a helpful assistant."
	dossier := &UserDossier{
		UserID:  "user1",
		Content: "- Data engineer\n- Budget: $5000",
	}

	result := BuildAugmentedSystemPrompt(basePrompt, dossier, nil)

	if !strings.Contains(result, basePrompt) {
		t.Error("Expected base prompt to be included")
	}
	if !strings.Contains(result, "USER DOSSIER") {
		t.Error("Expected dossier section")
	}
	if !strings.Contains(result, "Data engineer") {
		t.Error("Expected dossier content")
	}
	if strings.Contains(result, "RELEVANT USER FACTS") {
		t.Error("Should not include facts section when nil")
	}
}

func TestBuildAugmentedSystemPrompt_WithFacts(t *testing.T) {
	basePrompt := "You are a helpful assistant."
	facts := []*UserFact{
		{Category: "work", Fact: "Maintains a Go service called Ledger", Score: 0.95},
		{Category: "location", Fact: "Lives in Berlin", Score: 0.87},
	}

	result := BuildAugmentedSystemPrompt(basePrompt, nil, facts)

	if !strings.Contains(result, basePrompt) {
		t.Error("Expected base prompt to be included")
	}
	if !strings.Contains(result, "RELEVANT USER FACTS") {
		t.Error("Expected facts section")
	}
	if !strings.Contains(result, "Ledger") {
		t.Error("Expected fact content")
	}
	if !strings.Contains(result, "0.95") {
		t.Error("Expected fact score")
	}
}

func TestBuildAugmentedSystemPrompt_WithDossierAndFacts(t *testing.T) {
	basePrompt := "You are a helpful assistant."
	dossier := &UserDossier{
		UserID:  "user1",
		Content: "- Data engineer in Berlin",
	}
	facts := []*UserFact{
		{Category: "preference", Fact: "Prefers short answers", Score: 0.95},
	}

	result := BuildAugmentedSystemPrompt(basePrompt, dossier, facts)

	if !strings.Contains(result, basePrompt) {
		t.Error("Expected base prompt")
	}
	if !strings.Contains(result, "USER DOSSIER") {
		t.Error("Expected dossier section")
	}
	if !strings.Contains(result, "RELEVANT USER FACTS") {
		t.Error("Expected facts section")
	}

	// Order: base prompt, then dossier, then facts.
	baseIdx := strings.Index(result, basePrompt)
	dossierIdx := strings.Index(result, "USER DOSSIER")
	factsIdx := strings.Index(result, "RELEVANT USER FACTS")

	if baseIdx >= dossierIdx || dossierIdx >= factsIdx {
		t.Error("Expected order: base -> dossier -> facts")
	}
}

func TestBuildAugmentedSystemPrompt_EmptyDossier(t *testing.T) {
	basePrompt := "You are a helpful assistant."
	dossier := &UserDossier{
		UserID:  "user1",
		Content: "",
	}

	result := BuildAugmentedSystemPrompt(basePrompt, dossier, nil)

	if strings.Contains(result, "USER DOSSIER") {
		t.Error("Should not include dossier section when content is empty")
	}
}

func TestBuildAugmentedSystemPrompt_WhitespaceDossier(t *testing.T) {
	basePrompt := "You are a helpful assistant."
	dossier := &UserDossier{
		UserID:  "user1",
		Content: "   \n\t  ",
	}

	result := BuildAugmentedSystemPrompt(basePrompt, dossier, nil)

	if strings.Contains(result, "USER DOSSIER") {
		t.Error("Should not include dossier section when content is whitespace only")
	}
}
