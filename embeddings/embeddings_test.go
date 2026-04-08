package embeddings

import (
	"context"
	"testing"
)

type stubEmbedder struct{}

func (s *stubEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float64, error) {
	results := make([][]float64, len(texts))
	for i, text := range texts {
		results[i] = []float64{float64(len(text))}
	}
	return results, nil
}

func (s *stubEmbedder) EmbedQuery(_ context.Context, text string) ([]float64, error) {
	return []float64{float64(len(text))}, nil
}

func TestEmbedderContract(t *testing.T) {
	var embedder Embedder = &stubEmbedder{}

	docs, err := embedder.EmbedDocuments(context.Background(), []string{"a", "abcd"})
	if err != nil {
		t.Fatalf("EmbedDocuments error: %v", err)
	}
	if len(docs) != 2 || docs[0][0] != 1 || docs[1][0] != 4 {
		t.Fatalf("unexpected document embeddings: %#v", docs)
	}

	query, err := embedder.EmbedQuery(context.Background(), "abc")
	if err != nil {
		t.Fatalf("EmbedQuery error: %v", err)
	}
	if len(query) != 1 || query[0] != 3 {
		t.Fatalf("unexpected query embedding: %#v", query)
	}
}
