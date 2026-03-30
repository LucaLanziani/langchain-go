package inmemory

import (
	"context"
	"fmt"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

// mockEmbedder returns deterministic embeddings based on text content.
type mockEmbedder struct{}

func (m *mockEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	vecs := make([][]float64, len(texts))
	for i, t := range texts {
		vecs[i] = textToVec(t)
	}
	return vecs, nil
}

func (m *mockEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return textToVec(text), nil
}

// textToVec creates a simple deterministic 3-dim embedding.
func textToVec(text string) []float64 {
	v := []float64{0, 0, 0}
	for i, c := range text {
		v[i%3] += float64(c)
	}
	// Normalize.
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	if norm > 0 {
		norm = 1.0 / (norm * 0.5) // simple scaling
		for i := range v {
			v[i] *= norm
		}
	}
	return v
}

func TestAddDocumentsAndSearch(t *testing.T) {
	ctx := context.Background()
	store := New(&mockEmbedder{})

	docs := []*core.Document{
		{PageContent: "golang programming", Metadata: map[string]any{"source": "go"}},
		{PageContent: "python programming", Metadata: map[string]any{"source": "py"}},
		{PageContent: "javascript web", Metadata: map[string]any{"source": "js"}},
	}

	ids, err := store.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ids) != 3 {
		t.Errorf("expected 3 ids, got %d", len(ids))
	}

	results, err := store.SimilaritySearch(ctx, "golang programming", 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestAddDocumentsWithExistingID(t *testing.T) {
	ctx := context.Background()
	store := New(&mockEmbedder{})

	docs := []*core.Document{
		{ID: "doc-1", PageContent: "fixed id document"},
	}
	ids, err := store.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ids[0] != "doc-1" {
		t.Errorf("expected id 'doc-1', got %q", ids[0])
	}
}

func TestSimilaritySearchWithScore(t *testing.T) {
	ctx := context.Background()
	store := New(&mockEmbedder{})

	docs := []*core.Document{
		{PageContent: "abc"},
		{PageContent: "xyz"},
	}
	_, err := store.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	results, err := store.SimilaritySearchWithScore(ctx, "abc", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
	if results[0].Score < 0 || results[0].Score > 1.1 {
		t.Errorf("unexpected score %v", results[0].Score)
	}
}

func TestSimilaritySearchKGreaterThanDocs(t *testing.T) {
	ctx := context.Background()
	store := New(&mockEmbedder{})

	docs := []*core.Document{{PageContent: "only one doc"}}
	_, err := store.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// k=10 but only 1 doc.
	results, err := store.SimilaritySearch(ctx, "query", 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("expected 1 result (capped), got %d", len(results))
	}
}

func TestDelete(t *testing.T) {
	ctx := context.Background()
	store := New(&mockEmbedder{})

	docs := []*core.Document{
		{ID: "del-1", PageContent: "to delete"},
		{ID: "keep-1", PageContent: "to keep"},
	}
	_, err := store.AddDocuments(ctx, docs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := store.Delete(ctx, []string{"del-1"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	results, err := store.SimilaritySearch(ctx, "to keep", 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, r := range results {
		if r.ID == "del-1" {
			t.Error("expected del-1 to be deleted")
		}
	}
}

func TestGetEmbedder(t *testing.T) {
	e := &mockEmbedder{}
	store := New(e)
	if store.GetEmbedder() != e {
		t.Error("expected GetEmbedder to return the same embedder")
	}
}

func TestCosineSimilarityIdentical(t *testing.T) {
	v := []float64{1, 2, 3}
	sim := cosineSimilarity(v, v)
	// cos(v, v) = 1.0
	if sim < 0.999 {
		t.Errorf("expected ~1.0 for identical vectors, got %v", sim)
	}
}

func TestCosineSimilarityZero(t *testing.T) {
	v := []float64{0, 0, 0}
	sim := cosineSimilarity(v, v)
	if sim != 0 {
		t.Errorf("expected 0 for zero vectors, got %v", sim)
	}
}

func TestCosineSimilarityDifferentLength(t *testing.T) {
	a := []float64{1, 2}
	b := []float64{1, 2, 3}
	sim := cosineSimilarity(a, b)
	if sim != 0 {
		t.Errorf("expected 0 for different length vectors, got %v", sim)
	}
}

// failingEmbedder always returns an error.
type failingEmbedder struct{}

func (f *failingEmbedder) EmbedDocuments(_ context.Context, _ []string) ([][]float64, error) {
	return nil, fmt.Errorf("embed error")
}
func (f *failingEmbedder) EmbedQuery(_ context.Context, _ string) ([]float64, error) {
	return nil, fmt.Errorf("embed query error")
}

func TestAddDocumentsEmbedError(t *testing.T) {
	ctx := context.Background()
	store := New(&failingEmbedder{})
	_, err := store.AddDocuments(ctx, []*core.Document{{PageContent: "test"}})
	if err == nil {
		t.Error("expected error from failing embedder")
	}
}

func TestSimilaritySearchEmbedQueryError(t *testing.T) {
	ctx := context.Background()
	store := New(&failingEmbedder{})
	_, err := store.SimilaritySearch(ctx, "query", 1)
	if err == nil {
		t.Error("expected error from failing query embedder")
	}
}
