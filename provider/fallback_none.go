package provider

import (
	"context"

	"github.com/LucaLanziani/langchain-go/llms"
)

// GetFallbackProvider always returns an error (no fallback)
func (n *NoFallback) GetFallbackProvider(ctx context.Context, failedProvider string, providers map[string]llms.ChatModel) (string, error) {
	return "", ErrNoFallbackAvailable
}

// ShouldRetry always returns false (never retry)
func (n *NoFallback) ShouldRetry(err error, attemptCount int) bool {
	return false
}
