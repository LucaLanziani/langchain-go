// Package vectorstores provides interfaces and implementations for vector stores.
package vectorstores

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/embeddings"
)

// VectorStore is the interface for vector stores that support
// similarity search over embedded documents.
type VectorStore interface {
	// AddDocuments embeds and adds documents to the store.
	AddDocuments(ctx context.Context, documents []*core.Document) ([]string, error)

	// SimilaritySearch searches for documents similar to the query.
	SimilaritySearch(ctx context.Context, query string, k int) ([]*core.Document, error)

	// SimilaritySearchWithScore searches and returns documents with similarity scores.
	SimilaritySearchWithScore(ctx context.Context, query string, k int) ([]DocumentWithScore, error)

	// Delete removes documents by their IDs.
	Delete(ctx context.Context, ids []string) error

	// GetEmbedder returns the embedder used by this store.
	GetEmbedder() embeddings.Embedder
}

// DocumentWithScore pairs a document with its similarity score.
type DocumentWithScore struct {
	Document *core.Document
	Score    float64
}
