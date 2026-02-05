// Package inmemory provides an in-memory vector store implementation.
package inmemory

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/google/uuid"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/embeddings"
	"github.com/LucaLanziani/langchain-go/vectorstores"
)

type storedDoc struct {
	ID        string
	Document  *core.Document
	Embedding []float64
}

// Store is an in-memory vector store that uses cosine similarity.
type Store struct {
	embedder embeddings.Embedder
	docs     []storedDoc
	mu       sync.RWMutex
}

// New creates a new in-memory vector store.
func New(embedder embeddings.Embedder) *Store {
	return &Store{
		embedder: embedder,
	}
}

// AddDocuments embeds and stores documents.
func (s *Store) AddDocuments(ctx context.Context, documents []*core.Document) ([]string, error) {
	texts := make([]string, len(documents))
	for i, doc := range documents {
		texts[i] = doc.PageContent
	}

	vecs, err := s.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("failed to embed documents: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	ids := make([]string, len(documents))
	for i, doc := range documents {
		id := doc.ID
		if id == "" {
			id = uuid.New().String()
		}
		ids[i] = id
		s.docs = append(s.docs, storedDoc{
			ID:        id,
			Document:  doc,
			Embedding: vecs[i],
		})
	}

	return ids, nil
}

// SimilaritySearch finds the k most similar documents to the query.
func (s *Store) SimilaritySearch(ctx context.Context, query string, k int) ([]*core.Document, error) {
	results, err := s.SimilaritySearchWithScore(ctx, query, k)
	if err != nil {
		return nil, err
	}
	docs := make([]*core.Document, len(results))
	for i, r := range results {
		docs[i] = r.Document
	}
	return docs, nil
}

// SimilaritySearchWithScore finds the k most similar documents with scores.
func (s *Store) SimilaritySearchWithScore(ctx context.Context, query string, k int) ([]vectorstores.DocumentWithScore, error) {
	queryVec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	type scored struct {
		doc   *core.Document
		score float64
	}
	var scored_ []scored
	for _, d := range s.docs {
		sim := cosineSimilarity(queryVec, d.Embedding)
		scored_ = append(scored_, scored{doc: d.Document, score: sim})
	}

	// Sort by score descending.
	sort.Slice(scored_, func(i, j int) bool {
		return scored_[i].score > scored_[j].score
	})

	if k > len(scored_) {
		k = len(scored_)
	}

	results := make([]vectorstores.DocumentWithScore, k)
	for i := 0; i < k; i++ {
		results[i] = vectorstores.DocumentWithScore{
			Document: scored_[i].doc,
			Score:    scored_[i].score,
		}
	}
	return results, nil
}

// Delete removes documents by their IDs.
func (s *Store) Delete(_ context.Context, ids []string) error {
	idSet := make(map[string]bool, len(ids))
	for _, id := range ids {
		idSet[id] = true
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	filtered := s.docs[:0]
	for _, d := range s.docs {
		if !idSet[d.ID] {
			filtered = append(filtered, d)
		}
	}
	s.docs = filtered
	return nil
}

// GetEmbedder returns the embedder.
func (s *Store) GetEmbedder() embeddings.Embedder {
	return s.embedder
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Ensure Store implements vectorstores.VectorStore.
var _ vectorstores.VectorStore = (*Store)(nil)
