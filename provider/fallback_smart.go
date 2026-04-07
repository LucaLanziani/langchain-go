package provider

import (
	"context"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// GetFallbackProvider selects the best alternative provider based on metrics
// Never returns the failed provider
func (s *SmartFallback) GetFallbackProvider(ctx context.Context, failedProvider string, providers map[string]llms.ChatModel) (string, error) {
	if s.metrics == nil {
		return "", ErrNoFallbackAvailable
	}

	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	var bestProvider string
	var bestScore float64 = -1.0

	for providerName := range providers {
		// Skip the failed provider
		if providerName == failedProvider {
			continue
		}

		score := s.calculateProviderScore(providerName)

		if score > bestScore {
			bestScore = score
			bestProvider = providerName
		}
	}

	if bestProvider == "" {
		return "", ErrNoFallbackAvailable
	}

	return bestProvider, nil
}

// calculateProviderScore computes a score based on success rate and recency
// Higher score = better provider
func (s *SmartFallback) calculateProviderScore(providerName string) float64 {
	requestCount := s.metrics.RequestCount[providerName]
	errorCount := s.metrics.ErrorCount[providerName]
	lastUsed := s.metrics.LastUsed[providerName]

	// If never used, give it a neutral score
	if requestCount == 0 {
		return 0.5
	}

	// Calculate success rate (0.0 to 1.0)
	successRate := float64(requestCount-errorCount) / float64(requestCount)

	// Calculate recency bonus (0.0 to 0.2)
	// Providers used more recently get a small bonus
	recencyBonus := 0.0
	if !lastUsed.IsZero() {
		timeSinceLastUse := time.Since(lastUsed)

		// Give bonus if used within last hour
		if timeSinceLastUse < time.Hour {
			// Linear decay from 0.2 to 0.0 over 1 hour
			recencyBonus = 0.2 * (1.0 - float64(timeSinceLastUse)/float64(time.Hour))
		}
	}

	// Combined score: success rate (0.0-1.0) + recency bonus (0.0-0.2)
	return successRate + recencyBonus
}

// ShouldRetry returns true if the error is retryable and attempt count is reasonable
func (s *SmartFallback) ShouldRetry(err error, attemptCount int) bool {
	// Allow up to 3 retry attempts
	return attemptCount < 3
}
