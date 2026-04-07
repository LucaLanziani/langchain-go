package provider

import (
	"context"
	"math/rand"
	"sort"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// NewWeightedStrategy creates a new WeightedStrategy with the given weights.
// If a provider is not in the weights map, it defaults to weight 1.
func NewWeightedStrategy(weights map[string]int) *WeightedStrategy {
	return &WeightedStrategy{
		weights: weights,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SelectProvider returns a provider based on weighted random selection.
// The probability of selecting a provider is proportional to its weight.
func (s *WeightedStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if len(providers) == 0 {
		return "", ErrNoProvidersAvailable
	}

	// Fast path for single provider
	if len(providers) == 1 {
		for name := range providers {
			return name, nil
		}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	// Get provider names in deterministic order
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)

	// Calculate total weight
	totalWeight := 0
	for _, name := range names {
		weight := s.weights[name]
		if weight <= 0 {
			weight = 1 // Default weight
		}
		totalWeight += weight
	}

	// Select random provider based on weights
	random := s.rng.Intn(totalWeight)
	cumulative := 0

	for _, name := range names {
		weight := s.weights[name]
		if weight <= 0 {
			weight = 1
		}

		cumulative += weight

		if random < cumulative {
			return name, nil
		}
	}

	// Fallback (should never reach here)
	return names[0], nil
}

// OnSuccess is a no-op for WeightedStrategy.
func (s *WeightedStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: WeightedStrategy doesn't adapt based on feedback
}

// OnError is a no-op for WeightedStrategy.
func (s *WeightedStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: WeightedStrategy doesn't adapt based on feedback
}

// SetWeight updates the weight for a provider.
// This allows dynamic adjustment of routing weights.
func (s *WeightedStrategy) SetWeight(providerName string, weight int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.weights == nil {
		s.weights = make(map[string]int)
	}
	s.weights[providerName] = weight
}

// GetWeight returns the weight for a provider.
func (s *WeightedStrategy) GetWeight(providerName string) int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	weight := s.weights[providerName]
	if weight <= 0 {
		return 1 // Default weight
	}
	return weight
}
