package provider

import (
	"context"

	"github.com/LucaLanziani/langchain-go/llms"
)

// GetFallbackProvider returns the next provider in the configured order
// Never returns the failed provider
func (s *SequentialFallback) GetFallbackProvider(ctx context.Context, failedProvider string, providers map[string]llms.ChatModel) (string, error) {
	if len(s.Order) == 0 {
		return "", ErrNoFallbackAvailable
	}

	// Find the failed provider in the order
	failedIndex := -1
	for i, name := range s.Order {
		if name == failedProvider {
			failedIndex = i
			break
		}
	}

	// Try providers after the failed one
	startIndex := failedIndex + 1
	if failedIndex == -1 {
		// Failed provider not in order, start from beginning
		startIndex = 0
	}

	// Search for next available provider
	for i := startIndex; i < len(s.Order); i++ {
		providerName := s.Order[i]

		// Skip the failed provider
		if providerName == failedProvider {
			continue
		}

		// Check if provider exists
		if _, exists := providers[providerName]; exists {
			return providerName, nil
		}
	}

	// If we didn't find one after the failed provider, try from the beginning
	if failedIndex != -1 {
		for i := 0; i < failedIndex; i++ {
			providerName := s.Order[i]

			// Skip the failed provider
			if providerName == failedProvider {
				continue
			}

			// Check if provider exists
			if _, exists := providers[providerName]; exists {
				return providerName, nil
			}
		}
	}

	return "", ErrNoFallbackAvailable
}

// ShouldRetry returns true if there are more providers to try
func (s *SequentialFallback) ShouldRetry(err error, attemptCount int) bool {
	// Allow retries up to the number of providers in the order
	return attemptCount < len(s.Order)
}
