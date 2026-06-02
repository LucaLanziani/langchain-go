package provider

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// Mock provider for benchmarking
type mockProvider struct {
	name    string
	latency time.Duration
}

func (m *mockProvider) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	return core.NewAIMessage("response"), nil
}

func (m *mockProvider) Stream(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	return nil, nil
}

func (m *mockProvider) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	return nil, nil
}

func (m *mockProvider) GetName() string {
	return m.name
}

func (m *mockProvider) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	return &llms.ChatResult{
		Generations: []*llms.ChatGeneration{
			{Message: core.NewAIMessage("response")},
		},
	}, nil
}

func (m *mockProvider) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	return m
}

func (m *mockProvider) BindSkills(skills ...llms.SkillDefinition) llms.ChatModel {
	return m
}

func (m *mockProvider) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	return m
}

// ===== Provider Creation Benchmarks =====

func BenchmarkNewProvider_OpenAI(b *testing.B) {
	ctx := context.Background()
	opts := []ProviderOption{
		WithModel("gpt-4o"),
		WithAPIKey("test-key"),
		WithTemperature(0.7),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI, opts...)
		if err != nil {
			b.Fatal(err)
		}
		_ = cleanup()
		_ = model
	}
}

func BenchmarkNewProvider_Anthropic(b *testing.B) {
	ctx := context.Background()
	maxTokens := 1000
	opts := []ProviderOption{
		WithModel("claude-3-opus"),
		WithAPIKey("test-key"),
		WithMaxTokens(maxTokens),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic, opts...)
		if err != nil {
			b.Fatal(err)
		}
		_ = cleanup()
		_ = model
	}
}

// ===== Router Creation Benchmarks =====

func BenchmarkNewRouter_2Providers(b *testing.B) {
	benchmarkNewRouter(b, 2)
}

func BenchmarkNewRouter_5Providers(b *testing.B) {
	benchmarkNewRouter(b, 5)
}

func BenchmarkNewRouter_10Providers(b *testing.B) {
	benchmarkNewRouter(b, 10)
}

func benchmarkNewRouter(b *testing.B, numProviders int) {
	ctx := context.Background()

	entries := make([]ProviderEntry, numProviders)
	for i := 0; i < numProviders; i++ {
		entries[i] = ProviderEntry{
			Name:         "provider-" + string(rune('a'+i)),
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		}
	}

	strategy := &SimpleStrategy{ProviderName: entries[0].Name}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router, err := NewRouter(ctx, entries, strategy)
		if err != nil {
			b.Fatal(err)
		}
		_ = router.Cleanup()
	}
}

// ===== Routing Strategy Benchmarks =====

func BenchmarkSimpleStrategy_SelectProvider(b *testing.B) {
	providers := createMockProviders(5)
	strategy := &SimpleStrategy{ProviderName: "provider-0"}
	ctx := context.Background()
	reqCtx := RequestContext{MessageCount: 1, TotalTokens: 100}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRoundRobinStrategy_SelectProvider(b *testing.B) {
	providers := createMockProviders(5)
	strategy := &RoundRobinStrategy{}
	ctx := context.Background()
	reqCtx := RequestContext{MessageCount: 1, TotalTokens: 100}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWeightedStrategy_SelectProvider(b *testing.B) {
	providers := createMockProviders(5)
	weights := map[string]int{
		"provider-0": 5,
		"provider-1": 3,
		"provider-2": 2,
		"provider-3": 1,
		"provider-4": 1,
	}
	strategy := NewWeightedStrategy(weights)
	ctx := context.Background()
	reqCtx := RequestContext{MessageCount: 1, TotalTokens: 100}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRuleBasedStrategy_SelectProvider(b *testing.B) {
	providers := createMockProviders(5)
	rules := []RoutingRule{
		{
			Name:     "complex",
			Priority: 10,
			Provider: "provider-0",
			Condition: func(ctx RequestContext) bool {
				return ctx.Complexity == "complex"
			},
		},
		{
			Name:     "simple",
			Priority: 5,
			Provider: "provider-1",
			Condition: func(ctx RequestContext) bool {
				return ctx.Complexity == "simple"
			},
		},
	}
	strategy := NewRuleBasedStrategy(rules, "provider-2")
	ctx := context.Background()
	reqCtx := RequestContext{MessageCount: 1, TotalTokens: 100, Complexity: "simple"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLoadBalancedStrategy_SelectProvider(b *testing.B) {
	providers := createMockProviders(5)
	metrics := newRouterMetrics()

	// Populate metrics with sample data
	for name := range providers {
		metrics.RequestCount[name] = 100
		metrics.ErrorCount[name] = 5
		metrics.TotalLatency[name] = 10 * time.Second
		metrics.LastUsed[name] = time.Now()
	}

	strategy := NewLoadBalancedStrategy(metrics)
	ctx := context.Background()
	reqCtx := RequestContext{MessageCount: 1, TotalTokens: 100}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// ===== Concurrent Request Benchmarks =====

func BenchmarkRouter_ConcurrentRequests_10(b *testing.B) {
	benchmarkConcurrentRequests(b, 10)
}

func BenchmarkRouter_ConcurrentRequests_100(b *testing.B) {
	benchmarkConcurrentRequests(b, 100)
}

func BenchmarkRouter_ConcurrentRequests_1000(b *testing.B) {
	benchmarkConcurrentRequests(b, 1000)
}

func benchmarkConcurrentRequests(b *testing.B, concurrency int) {
	router := createMockRouter(5)
	defer router.Cleanup()

	ctx := context.Background()
	messages := []core.Message{
		core.NewHumanMessage("test message"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		for j := 0; j < concurrency; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = router.Invoke(ctx, messages)
			}()
		}
		wg.Wait()
	}
}

// ===== Metrics Update Benchmarks =====

func BenchmarkMetrics_Update(b *testing.B) {
	metrics := newRouterMetrics()
	providerName := "test-provider"
	latency := 100 * time.Millisecond

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metrics.mu.Lock()
		metrics.RequestCount[providerName]++
		metrics.TotalLatency[providerName] += latency
		metrics.LastUsed[providerName] = time.Now()
		metrics.mu.Unlock()
	}
}

func BenchmarkMetrics_GetStats(b *testing.B) {
	metrics := newRouterMetrics()
	providerName := "test-provider"

	// Populate with sample data
	metrics.RequestCount[providerName] = 1000
	metrics.ErrorCount[providerName] = 50
	metrics.TotalLatency[providerName] = 100 * time.Second
	metrics.LastUsed[providerName] = time.Now()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = metrics.GetStats(providerName)
	}
}

func BenchmarkMetrics_GetAllStats(b *testing.B) {
	metrics := newRouterMetrics()

	// Populate with sample data for 10 providers
	for i := 0; i < 10; i++ {
		name := "provider-" + string(rune('0'+i))
		metrics.RequestCount[name] = 1000
		metrics.ErrorCount[name] = 50
		metrics.TotalLatency[name] = 100 * time.Second
		metrics.LastUsed[name] = time.Now()
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = metrics.GetAllStats()
	}
}

func BenchmarkMetrics_ConcurrentUpdates(b *testing.B) {
	metrics := newRouterMetrics()
	providerNames := []string{"p1", "p2", "p3", "p4", "p5"}
	latency := 100 * time.Millisecond

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			providerName := providerNames[i%len(providerNames)]
			metrics.mu.Lock()
			metrics.RequestCount[providerName]++
			metrics.TotalLatency[providerName] += latency
			metrics.LastUsed[providerName] = time.Now()
			metrics.mu.Unlock()
			i++
		}
	})
}

// ===== Request Context Building Benchmarks =====

func BenchmarkBuildRequestContext_Simple(b *testing.B) {
	messages := []core.Message{
		core.NewHumanMessage("Hello"),
	}
	opts := []core.Option{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = buildRequestContext(messages, opts)
	}
}

func BenchmarkBuildRequestContext_Complex(b *testing.B) {
	messages := []core.Message{
		core.NewHumanMessage("This is a much longer message with more content to process"),
		core.NewAIMessageWithToolCalls("Response with tool calls", []core.ToolCall{{ID: "1", Name: "tool"}}),
		core.NewToolMessage("Tool result", "1"),
		core.NewHumanMessage("Follow up question"),
	}
	opts := []core.Option{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = buildRequestContext(messages, opts)
	}
}

// ===== Router Invoke Benchmarks =====

func BenchmarkRouter_Invoke_SimpleStrategy(b *testing.B) {
	router := createMockRouter(5)
	defer router.Cleanup()

	ctx := context.Background()
	messages := []core.Message{
		core.NewHumanMessage("test message"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := router.Invoke(ctx, messages)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRouter_Invoke_RoundRobinStrategy(b *testing.B) {
	router := createMockRouterWithStrategy(5, &RoundRobinStrategy{})
	defer router.Cleanup()

	ctx := context.Background()
	messages := []core.Message{
		core.NewHumanMessage("test message"),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := router.Invoke(ctx, messages)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// ===== Helper Functions =====

func createMockProviders(count int) map[string]llms.ChatModel {
	providers := make(map[string]llms.ChatModel, count)
	for i := 0; i < count; i++ {
		name := "provider-" + string(rune('0'+i))
		providers[name] = &mockProvider{name: name, latency: 0}
	}
	return providers
}

func createMockRouter(numProviders int) *Router {
	providers := createMockProviders(numProviders)
	metrics := newRouterMetrics()

	// Initialize metrics
	for name := range providers {
		metrics.RequestCount[name] = 0
		metrics.ErrorCount[name] = 0
		metrics.TotalLatency[name] = 0
	}

	cleanups := make(map[string]CleanupFunc)
	for name := range providers {
		cleanups[name] = func() error { return nil }
	}

	return &Router{
		providers: providers,
		cleanups:  cleanups,
		strategy:  &SimpleStrategy{ProviderName: "provider-0"},
		metrics:   metrics,
	}
}

func createMockRouterWithStrategy(numProviders int, strategy RoutingStrategy) *Router {
	providers := createMockProviders(numProviders)
	metrics := newRouterMetrics()

	// Initialize metrics
	for name := range providers {
		metrics.RequestCount[name] = 0
		metrics.ErrorCount[name] = 0
		metrics.TotalLatency[name] = 0
	}

	cleanups := make(map[string]CleanupFunc)
	for name := range providers {
		cleanups[name] = func() error { return nil }
	}

	return &Router{
		providers: providers,
		cleanups:  cleanups,
		strategy:  strategy,
		metrics:   metrics,
	}
}
