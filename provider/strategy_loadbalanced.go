package provider

import (
	"context"
	"math"
	"sort"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// NewLoadBalancedStrategy creates a new LoadBalancedStrategy.
func NewLoadBalancedStrategy(metrics *RouterMetrics) *LoadBalancedStrategy {
	return &LoadBalancedStrategy{
		metrics: metrics,
	}
}

// SelectProvider returns the provider with the best score.
// Score is calculated based on latency, error rate, and load.
func (s *LoadBalancedStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if len(providers) == 0 {
		return "", ErrNoProvidersAvailable
	}

	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	// Get provider names in deterministic order
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)

	// Calculate scores for each provider
	type providerScore struct {
		name  string
		score float64
	}
	scores := make([]providerScore, 0, len(names))

	for _, name := range names {
		score := s.calculateScore(name)
		scores = append(scores, providerScore{name: name, score: score})
	}

	// Sort by score (higher is better)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Return provider with best score
	return scores[0].name, nil
}

// calculateScore computes a score for a provider based on metrics.
// Higher score = better provider to use.
func (s *LoadBalancedStrategy) calculateScore(providerName string) float64 {
	requestCount := s.metrics.RequestCount[providerName]
	errorCount := s.metrics.ErrorCount[providerName]
	totalLatency := s.metrics.TotalLatency[providerName]
	lastUsed := s.metrics.LastUsed[providerName]

	// If provider has never been used, give it high priority
	if requestCount == 0 {
		return 1000.0
	}

	// Calculate average latency in milliseconds
	avgLatency := float64(totalLatency.Milliseconds()) / float64(requestCount)

	// Calculate error rate (0.0 to 1.0)
	errorRate := float64(errorCount) / float64(requestCount)

	// Calculate recency penalty (prefer less recently used providers)
	timeSinceLastUse := time.Since(lastUsed).Seconds()
	recencyBonus := math.Min(timeSinceLastUse/60.0, 1.0) // Max bonus at 1 minute

	// Calculate score (higher is better)
	// - Lower latency = higher score
	// - Lower error rate = higher score
	// - Less recent usage = higher score (load balancing)
	latencyScore := 1000.0 / (avgLatency + 1.0) // Avoid division by zero
	errorScore := (1.0 - errorRate) * 100.0     // 0-100 range
	recencyScore := recencyBonus * 50.0         // 0-50 range

	totalScore := latencyScore + errorScore + recencyScore

	return totalScore
}

// OnSuccess is a no-op for LoadBalancedStrategy.
// Metrics are updated by the router, so no additional action needed.
func (s *LoadBalancedStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: Metrics are updated by the router
}

// OnError is a no-op for LoadBalancedStrategy.
// Metrics are updated by the router, so no additional action needed.
func (s *LoadBalancedStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: Metrics are updated by the router
}
