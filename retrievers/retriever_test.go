package retrievers

import (
	"context"
	"fmt"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/embeddings"
	"github.com/LucaLanziani/langchain-go/vectorstores"
)

// mockVectorStore implements vectorstores.VectorStore for testing.
type mockVectorStore struct {
	docs []*core.Document
	err  error
}

func (m *mockVectorStore) AddDocuments(_ context.Context, docs []*core.Document) ([]string, error) {
	return nil, nil
}

func (m *mockVectorStore) SimilaritySearch(_ context.Context, query string, k int) ([]*core.Document, error) {
	if m.err != nil {
		return nil, m.err
	}
	if k > len(m.docs) {
		k = len(m.docs)
	}
	return m.docs[:k], nil
}

func (m *mockVectorStore) SimilaritySearchWithScore(_ context.Context, query string, k int) ([]vectorstores.DocumentWithScore, error) {
	return nil, nil
}

func (m *mockVectorStore) Delete(_ context.Context, ids []string) error {
	return nil
}

func (m *mockVectorStore) GetEmbedder() embeddings.Embedder {
	return nil
}

func TestNewVectorStoreRetriever(t *testing.T) {
	store := &mockVectorStore{
		docs: []*core.Document{
			{PageContent: "doc1"},
			{PageContent: "doc2"},
		},
	}
	r := NewVectorStoreRetriever(store, 2)
	if r.GetName() != "VectorStoreRetriever" {
		t.Errorf("expected 'VectorStoreRetriever', got %q", r.GetName())
	}
}

func TestVectorStoreRetrieverDefaultK(t *testing.T) {
	store := &mockVectorStore{}
	r := NewVectorStoreRetriever(store, 0) // 0 → default 4
	if r.k != 4 {
		t.Errorf("expected default k=4, got %d", r.k)
	}
}

func TestVectorStoreRetrieverWithName(t *testing.T) {
	store := &mockVectorStore{}
	r := NewVectorStoreRetriever(store, 3).WithName("MyRetriever")
	if r.GetName() != "MyRetriever" {
		t.Errorf("expected 'MyRetriever', got %q", r.GetName())
	}
}

func TestVectorStoreRetrieverInvoke(t *testing.T) {
	store := &mockVectorStore{
		docs: []*core.Document{
			{PageContent: "relevant doc"},
		},
	}
	r := NewVectorStoreRetriever(store, 1)
	docs, err := r.Invoke(context.Background(), "query")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(docs) != 1 || docs[0].PageContent != "relevant doc" {
		t.Errorf("unexpected docs: %v", docs)
	}
}

func TestVectorStoreRetrieverStream(t *testing.T) {
	store := &mockVectorStore{
		docs: []*core.Document{
			{PageContent: "doc1"},
		},
	}
	r := NewVectorStoreRetriever(store, 1)
	iter, err := r.Stream(context.Background(), "query")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	docs, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if len(docs) != 1 {
		t.Errorf("expected 1 doc, got %d", len(docs))
	}
}

func TestVectorStoreRetrieverBatch(t *testing.T) {
	store := &mockVectorStore{
		docs: []*core.Document{
			{PageContent: "doc1"},
		},
	}
	r := NewVectorStoreRetriever(store, 1)
	results, err := r.Batch(context.Background(), []string{"q1", "q2"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestVectorStoreRetrieverError(t *testing.T) {
	store := &mockVectorStore{err: fmt.Errorf("search error")}
	r := NewVectorStoreRetriever(store, 1)
	_, err := r.Invoke(context.Background(), "query")
	if err == nil {
		t.Error("expected error from failing store")
	}
}

func TestVectorStoreRetrieverStreamError(t *testing.T) {
	store := &mockVectorStore{err: fmt.Errorf("stream error")}
	r := NewVectorStoreRetriever(store, 1)
	_, err := r.Stream(context.Background(), "query")
	if err == nil {
		t.Error("expected error from failing store in stream")
	}
}

func TestVectorStoreRetrieverBatchError(t *testing.T) {
	store := &mockVectorStore{err: fmt.Errorf("batch search error")}
	r := NewVectorStoreRetriever(store, 1)
	_, err := r.Batch(context.Background(), []string{"q1", "q2"})
	if err == nil {
		t.Error("expected error from failing store in batch")
	}
}
