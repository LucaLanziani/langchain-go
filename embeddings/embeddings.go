// Package embeddings provides the interface for text embedding models.
package embeddings

import (
	"context"
)

// Embedder is the interface for text embedding models.
// Embeddings convert text into dense vector representations for
// similarity search, clustering, and retrieval.
type Embedder interface {
	// EmbedDocuments embeds multiple texts.
	EmbedDocuments(ctx context.Context, texts []string) ([][]float64, error)

	// EmbedQuery embeds a single query text.
	// Some models distinguish between document and query embeddings.
	EmbedQuery(ctx context.Context, text string) ([]float64, error)
}
