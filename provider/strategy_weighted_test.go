package provider

import (
	"context"
	"sync"
	"testing"

	"github.com/LucaLanziani/langchain-go/llms"
)

func TestWeightedStrategyConcurrentSelectProvider(t *testing.T) {
	strategy := NewWeightedStrategy(map[string]int{"openai": 3, "anthropic": 1})
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	var wg sync.WaitGroup
	errs := make(chan error, 100)
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			providerName, err := strategy.SelectProvider(context.Background(), RequestContext{}, providers)
			if err != nil {
				errs <- err
				return
			}
			if _, ok := providers[providerName]; !ok {
				errs <- ErrProviderNotFound
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatalf("unexpected concurrent selection error: %v", err)
	}
}

func TestWeightedStrategyPreservesWeightBias(t *testing.T) {
	strategy := NewWeightedStrategy(map[string]int{"openai": 10, "anthropic": 1})
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	counts := map[string]int{}
	for i := 0; i < 5000; i++ {
		providerName, err := strategy.SelectProvider(context.Background(), RequestContext{}, providers)
		if err != nil {
			t.Fatalf("SelectProvider error: %v", err)
		}
		counts[providerName]++
	}

	if counts["openai"] <= counts["anthropic"] {
		t.Fatalf("expected weighted selection to favor openai, got counts=%v", counts)
	}
}
