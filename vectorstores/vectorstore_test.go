package vectorstores_test

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/embeddings"
	"github.com/LucaLanziani/langchain-go/vectorstores"
	"github.com/LucaLanziani/langchain-go/vectorstores/inmemory"
)

type stubEmbedder struct{}

func (s *stubEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	results := make([][]float64, len(texts))
	for i := range texts {
		results[i] = []float64{float64(i + 1), 1}
	}
	return results, nil
}

func (s *stubEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return []float64{1, 1}, nil
}

func TestVectorStoreContract(t *testing.T) {
	var embedder embeddings.Embedder = &stubEmbedder{}
	var store vectorstores.VectorStore = inmemory.New(embedder)

	docs := []*core.Document{{PageContent: "alpha"}, {PageContent: "beta"}}
	ids, err := store.AddDocuments(context.Background(), docs)
	if err != nil {
		t.Fatalf("AddDocuments error: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 ids, got %d", len(ids))
	}

	results, err := store.SimilaritySearch(context.Background(), "query", 2)
	if err != nil {
		t.Fatalf("SimilaritySearch error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 search results, got %d", len(results))
	}

	if err := store.Delete(context.Background(), []string{ids[0]}); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
	results, err = store.SimilaritySearch(context.Background(), "query", 2)
	if err != nil {
		t.Fatalf("SimilaritySearch error after delete: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 remaining document after delete, got %d", len(results))
	}
}
