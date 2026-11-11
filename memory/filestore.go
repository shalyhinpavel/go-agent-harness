package memory

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"gopkg.in/yaml.v3"

	"github.com/google/uuid"

	"github.com/shalyhinpavel/go-agent-harness/internal/safename"
)

// FileStore keeps memory as plain markdown notes — one file per fact, a
// dossier file per user — inside a directory tree:
//
//	<dir>/<user>/fact-<id>.md   one note per fact (YAML frontmatter + body)
//	<dir>/<user>/dossier.md     the user's hot profile
//
// Notes are human-editable: edit the file and the next search sees it.
// Search is lexical by default (body token overlap, tag and category
// boosts); when the query carries an embedding and the fact has a stored
// one, cosine similarity takes over — vector search stays a port (an
// external embedder through NewSearchTool's embeddingFn), no embedder
// ships with the harness.
type FileStore struct {
	dir string
}

// NewFileStore creates a store rooted at dir (created if missing).
func NewFileStore(dir string) (*FileStore, error) {
	if dir == "" {
		return nil, fmt.Errorf("memory: file store needs a directory")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("memory: create store dir: %w", err)
	}
	return &FileStore{dir: dir}, nil
}

// noteFrontmatter is the YAML block of a fact note.
type noteFrontmatter struct {
	ID         string    `yaml:"id"`
	UserID     string    `yaml:"user_id"`
	Category   string    `yaml:"category"`
	Tags       []string  `yaml:"tags,omitempty"`
	Confidence float64   `yaml:"confidence"`
	CreatedAt  time.Time `yaml:"created_at"`
	UpdatedAt  time.Time `yaml:"updated_at"`
}

// Path helpers — every path stays inside the store dir even for hostile
// user or fact ids.
func (s *FileStore) userDir(userID string) string {
	return filepath.Join(s.dir, sanitizeID(userID))
}

func (s *FileStore) factPath(userID, factID string) string {
	return filepath.Join(s.userDir(userID), "fact-"+sanitizeID(factID)+".md")
}

func (s *FileStore) vectorPath(userID, factID string) string {
	return filepath.Join(s.userDir(userID), "fact-"+sanitizeID(factID)+".vec.yaml")
}

func (s *FileStore) dossierPath(userID string) string {
	return filepath.Join(s.userDir(userID), "dossier.md")
}

// writeFile lands content atomically: a crash never leaves half a note.
func writeFile(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	// A unique temp name: two writers of one note must not share a file.
	f, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	_, werr := f.WriteString(content)
	if cerr := f.Close(); werr == nil {
		werr = cerr
	}
	if werr == nil {
		werr = os.Chmod(tmp, 0o644)
	}
	if werr == nil {
		werr = os.Rename(tmp, path)
	}
	if werr != nil {
		_ = os.Remove(tmp)
	}
	return werr
}

// splitNote divides a markdown note into frontmatter head and body.
func splitNote(raw []byte) (head, body string, ok bool) {
	const delim = "---\n"
	// Notes edited on Windows or by hand carry CRLF line ends.
	text := strings.ReplaceAll(string(raw), "\r\n", "\n")
	if !strings.HasPrefix(text, delim) {
		return "", "", false
	}
	end := strings.Index(text[len(delim):], "\n"+delim)
	if end < 0 {
		return "", "", false
	}
	return text[len(delim) : len(delim)+end], strings.TrimSpace(text[len(delim)+end+len("\n"+delim):]), true
}

// InsertFact writes the fact as a markdown note (and its embedding to a
// sidecar file when present). An existing id is an update.
func (s *FileStore) InsertFact(_ context.Context, fact *UserFact) error {
	if fact == nil {
		return fmt.Errorf("memory: nil fact")
	}
	if fact.ID == "" {
		fact.ID = uuid.New().String()
	}
	if fact.CreatedAt.IsZero() {
		fact.CreatedAt = time.Now().UTC()
	}
	fact.UpdatedAt = time.Now().UTC()

	fm := noteFrontmatter{
		ID:         fact.ID,
		UserID:     fact.UserID,
		Category:   fact.Category,
		Tags:       fact.Tags,
		Confidence: fact.Confidence,
		CreatedAt:  fact.CreatedAt,
		UpdatedAt:  fact.UpdatedAt,
	}
	head, err := yaml.Marshal(fm)
	if err != nil {
		return fmt.Errorf("memory: marshal frontmatter: %w", err)
	}
	note := fmt.Sprintf("---\n%s---\n\n%s\n", head, strings.TrimSpace(fact.Fact))
	if err := writeFile(s.factPath(fact.UserID, fact.ID), note); err != nil {
		return fmt.Errorf("memory: write fact: %w", err)
	}

	// Embeddings live in a sidecar so notes stay human-readable.
	vecPath := s.vectorPath(fact.UserID, fact.ID)
	if len(fact.Embedding) > 0 {
		raw, err := yaml.Marshal(fact.Embedding)
		if err != nil {
			return fmt.Errorf("memory: marshal embedding: %w", err)
		}
		if err := writeFile(vecPath, string(raw)); err != nil {
			return fmt.Errorf("memory: write embedding: %w", err)
		}
	} else if _, err := os.Stat(vecPath); err == nil {
		_ = os.Remove(vecPath) // the update dropped the embedding
	}
	return nil
}

// loadFacts reads every fact note of a user from disk.
func (s *FileStore) loadFacts(userID string) ([]*UserFact, error) {
	entries, err := os.ReadDir(s.userDir(userID))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("memory: read user dir: %w", err)
	}

	var facts []*UserFact
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "fact-") || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(s.userDir(userID), e.Name()))
		if err != nil {
			continue // unreadable note — skip it, don't fail the search
		}
		head, body, ok := splitNote(raw)
		if !ok {
			continue // not a note — ignore
		}
		var fm noteFrontmatter
		if err := yaml.Unmarshal([]byte(head), &fm); err != nil {
			continue // broken frontmatter — ignore
		}
		facts = append(facts, &UserFact{
			ID:         fm.ID,
			UserID:     userID,
			Category:   fm.Category,
			Tags:       fm.Tags,
			Fact:       body,
			Confidence: fm.Confidence,
			CreatedAt:  fm.CreatedAt,
			UpdatedAt:  fm.UpdatedAt,
		})
	}
	return facts, nil
}

// SearchFacts scores the user's notes for the query and returns the top
// matches. Cosine similarity applies when both the query and the fact
// carry embeddings; otherwise lexical overlap. Notes that match nothing
// are not returned.
func (s *FileStore) SearchFacts(_ context.Context, userID, query string, embedding []float32, limit int) ([]*UserFact, error) {
	facts, err := s.loadFacts(userID)
	if err != nil {
		return nil, err
	}
	if len(facts) == 0 {
		return nil, nil
	}
	if limit <= 0 {
		limit = 5
	}

	results := make([]*UserFact, 0, len(facts))
	for _, f := range facts {
		var sim float64
		if len(embedding) > 0 {
			if vec, err := s.loadVector(userID, f.ID); err == nil && len(vec) > 0 {
				sim = cosine(vec, embedding)
			}
		}
		if sim == 0 {
			sim = lexicalScore(query, f.Fact, f.Category, f.Tags)
		}
		if sim <= 0 {
			continue
		}
		scored := *f
		scored.Score = CalculateScore(sim, f.CreatedAt, 30.0)
		results = append(results, &scored)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

// loadVector reads the sidecar embedding of a fact.
func (s *FileStore) loadVector(userID, factID string) ([]float32, error) {
	raw, err := os.ReadFile(s.vectorPath(userID, factID))
	if err != nil {
		return nil, err
	}
	var vec []float32
	if err := yaml.Unmarshal(raw, &vec); err != nil {
		return nil, err
	}
	return vec, nil
}

// DeleteFact removes the note (and its embedding sidecar).
func (s *FileStore) DeleteFact(_ context.Context, userID, factID string) error {
	removed := false
	for _, path := range []string{s.factPath(userID, factID), s.vectorPath(userID, factID)} {
		if err := os.Remove(path); err == nil {
			removed = true
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("memory: delete fact: %w", err)
		}
	}
	if !removed {
		return fmt.Errorf("memory: fact %s not found for user %s", factID, userID)
	}
	return nil
}

// SaveDossier writes the user's profile note.
func (s *FileStore) SaveDossier(_ context.Context, dossier *UserDossier) error {
	if dossier == nil {
		return fmt.Errorf("memory: nil dossier")
	}
	dossier.UpdatedAt = time.Now().UTC()

	head, err := yaml.Marshal(map[string]any{
		"user_id":    dossier.UserID,
		"updated_at": dossier.UpdatedAt,
	})
	if err != nil {
		return fmt.Errorf("memory: marshal dossier: %w", err)
	}
	note := fmt.Sprintf("---\n%s---\n\n%s\n", head, strings.TrimSpace(dossier.Content))
	if err := writeFile(s.dossierPath(dossier.UserID), note); err != nil {
		return fmt.Errorf("memory: write dossier: %w", err)
	}
	return nil
}

// GetDossier returns the user's profile note (empty content when missing).
func (s *FileStore) GetDossier(_ context.Context, userID string) (*UserDossier, error) {
	raw, err := os.ReadFile(s.dossierPath(userID))
	if err != nil {
		if os.IsNotExist(err) {
			return &UserDossier{UserID: userID, Content: ""}, nil
		}
		return nil, fmt.Errorf("memory: read dossier: %w", err)
	}

	head, body, ok := splitNote(raw)
	if !ok {
		return &UserDossier{UserID: userID, Content: strings.TrimSpace(string(raw))}, nil
	}
	var fm struct {
		UpdatedAt time.Time `yaml:"updated_at"`
	}
	_ = yaml.Unmarshal([]byte(head), &fm)
	return &UserDossier{UserID: userID, Content: body, UpdatedAt: fm.UpdatedAt}, nil
}

// lexicalScore computes the normalized overlap of query tokens with the
// fact body; tag matches add a quarter point each and an exact category
// token half a point. The result caps at 1.
func lexicalScore(query, body, category string, tags []string) float64 {
	q := tokenize(query)
	if len(q) == 0 {
		return 0
	}
	b := tokenize(body)

	overlap := 0
	for t := range q {
		if b[t] {
			overlap++
		}
	}
	score := float64(overlap) / float64(len(q))

	for t := range q {
		for _, tag := range tags {
			if tokenize(tag)[t] {
				score += 0.25
			}
		}
	}
	for t := range tokenize(category) {
		if q[t] {
			score += 0.5
		}
	}
	return math.Min(score, 1.0)
}

// tokenize splits lowercased text into letter-and-digit tokens in any
// script; one-letter tokens are dropped.
func tokenize(text string) map[string]bool {
	tokens := map[string]bool{}
	for _, t := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if utf8.RuneCountInString(t) > 1 {
			tokens[t] = true
		}
	}
	return tokens
}

// cosine is the similarity between two equal-length vectors (anything
// else scores 0).
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// sanitizeID keeps a user/fact id inside the store directory.
func sanitizeID(id string) string { return safename.Of(id, "anonymous") }
