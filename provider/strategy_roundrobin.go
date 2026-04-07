package provider

import (
	"context"
	"sort"
	"sync/atomic"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// SelectProvider returns providers in round-robin order.
// The distribution is guaranteed to be even across all providers over time.
func (s *RoundRobinStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if len(providers) == 0 {
		return "", ErrNoProvidersAvailable
	}

	// Fast path for single provider
	if len(providers) == 1 {
		for name := range providers {
			return name, nil
		}
	}

	// Get provider names in deterministic order
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)

	// Atomically increment counter and select provider
	index := atomic.AddUint64(&s.counter, 1) - 1
	selectedIndex := int(index % uint64(len(names)))

	return names[selectedIndex], nil
}

// OnSuccess is a no-op for RoundRobinStrategy.
func (s *RoundRobinStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op: RoundRobinStrategy doesn't adapt based on feedback
}

// OnError is a no-op for RoundRobinStrategy.
func (s *RoundRobinStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op: RoundRobinStrategy doesn't adapt based on feedback
}
