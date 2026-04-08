package provider

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// NewRouter creates a router that manages multiple providers.
// It validates that all provider entries have unique names and creates
// all providers using the factory. If any provider fails to initialize,
// all successfully created providers are cleaned up before returning an error.
//
// The router implements the llms.ChatModel interface, allowing it to be used
// transparently in place of a single provider.
//
// Example:
//
//	router, err := provider.NewRouter(ctx,
//		[]provider.ProviderEntry{
//			{Name: "openai", ProviderType: provider.ProviderOpenAI, Options: []provider.ProviderOption{provider.WithModel("gpt-4o")}},
//			{Name: "anthropic", ProviderType: provider.ProviderAnthropic, Options: []provider.ProviderOption{provider.WithModel("claude-3-opus")}},
//		},
//		&provider.SimpleStrategy{ProviderName: "openai"},
//	)
//	if err != nil {
//		return err
//	}
//	defer router.Cleanup()
func NewRouter(ctx context.Context, entries []ProviderEntry, strategy RoutingStrategy, opts ...RouterOption) (*Router, error) {
	// Validate entries
	if len(entries) == 0 {
		return nil, ErrEmptyProviderList
	}

	// Check for duplicate names
	names := make(map[string]bool)
	for _, entry := range entries {
		if names[entry.Name] {
			return nil, fmt.Errorf("%w: %s", ErrDuplicateProviderName, entry.Name)
		}
		names[entry.Name] = true
	}

	// Validate strategy
	if strategy == nil {
		return nil, fmt.Errorf("routing strategy is required")
	}

	// Apply router options
	config := &RouterConfig{
		EnableMetrics: true,
		MaxRetries:    3,
		RetryDelay:    100 * time.Millisecond,
	}
	for _, opt := range opts {
		opt(config)
	}

	// Initialize router
	router := &Router{
		providers: make(map[string]llms.ChatModel),
		cleanups:  make(map[string]CleanupFunc),
		strategy:  strategy,
		fallback:  config.FallbackStrategy,
		metrics:   newRouterMetrics(),
	}

	// Track created providers for cleanup on error
	createdProviders := make([]string, 0)

	// Create all providers
	// Note: No mutex needed during initialization as router is not yet shared
	for _, entry := range entries {
		model, cleanup, err := NewProvider(ctx, entry.ProviderType, entry.Options...)
		if err != nil {
			// Cleanup all successfully created providers
			for _, name := range createdProviders {
				if cleanupFn := router.cleanups[name]; cleanupFn != nil {
					_ = cleanupFn()
				}
			}
			return nil, NewProviderError(entry.ProviderType, entry.Name, "router_initialization", err)
		}

		router.providers[entry.Name] = model
		router.cleanups[entry.Name] = cleanup
		createdProviders = append(createdProviders, entry.Name)

		// Initialize metrics for this provider
		router.metrics.mu.Lock()
		router.metrics.RequestCount[entry.Name] = 0
		router.metrics.ErrorCount[entry.Name] = 0
		router.metrics.TotalLatency[entry.Name] = 0
		router.metrics.mu.Unlock()
	}

	return router, nil
}

// getCleanup returns the cleanup function for a provider by name.
// This is used during partial initialization failure cleanup.
// Note: This method is only called during initialization before the router
// is shared, so it doesn't need mutex protection.
func (r *Router) getCleanup(name string) CleanupFunc {
	return r.cleanups[name]
}

// Cleanup releases all resources held by the router and its providers.
// It calls the cleanup function for every provider it manages.
//
// Cleanup is idempotent - it can be called multiple times safely.
// If called while requests are in flight, those requests may fail.
//
// Returns the first error encountered during cleanup, but continues
// cleaning up remaining providers.
func (r *Router) Cleanup() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if already cleaned up
	if r.providers == nil {
		return nil
	}

	var firstErr error

	// Call cleanup for all providers
	for name, cleanup := range r.cleanups {
		if cleanup != nil {
			if err := cleanup(); err != nil && firstErr == nil {
				firstErr = fmt.Errorf("cleanup failed for provider %s: %w", name, err)
			}
		}
	}

	// Clear providers and cleanups maps to mark as cleaned up
	r.providers = nil
	r.cleanups = nil

	return firstErr
}

// newRouterMetrics creates a new RouterMetrics instance with initialized maps
func newRouterMetrics() *RouterMetrics {
	return &RouterMetrics{
		RequestCount:   make(map[string]int64),
		ErrorCount:     make(map[string]int64),
		CancelledCount: make(map[string]int64),
		TotalLatency:   make(map[string]time.Duration),
		LastUsed:       make(map[string]time.Time),
	}
}

// updateMetrics updates the metrics for a provider after a request
func (r *Router) updateMetrics(providerName string, latency time.Duration, isError bool) {
	r.metrics.mu.Lock()
	defer r.metrics.mu.Unlock()

	r.metrics.RequestCount[providerName]++
	r.metrics.TotalLatency[providerName] += latency
	r.metrics.LastUsed[providerName] = time.Now()

	if isError {
		r.metrics.ErrorCount[providerName]++
	}
}

func (r *Router) updateStreamMetrics(providerName string, latency time.Duration, isError bool, isCancelled bool) {
	r.metrics.mu.Lock()
	defer r.metrics.mu.Unlock()

	r.metrics.RequestCount[providerName]++
	r.metrics.TotalLatency[providerName] += latency
	r.metrics.LastUsed[providerName] = time.Now()

	if isCancelled {
		r.metrics.CancelledCount[providerName]++
		return
	}
	if isError {
		r.metrics.ErrorCount[providerName]++
	}
}

// GetProvider returns the ChatModel for a specific provider by name.
// Returns nil if the provider doesn't exist or the router has been cleaned up.
func (r *Router) GetProvider(name string) llms.ChatModel {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.providers == nil {
		return nil
	}

	return r.providers[name]
}

// ListProviders returns the names of all providers managed by this router.
// Returns nil if the router has been cleaned up.
func (r *Router) ListProviders() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.providers == nil {
		return nil
	}

	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// GetMetrics returns a copy of the current routing metrics.
// Returns nil if the router has been cleaned up.
func (r *Router) GetMetrics() map[string]ProviderMetrics {
	r.metrics.mu.RLock()
	defer r.metrics.mu.RUnlock()

	result := make(map[string]ProviderMetrics)
	for name := range r.metrics.RequestCount {
		result[name] = ProviderMetrics{
			RequestCount:   r.metrics.RequestCount[name],
			ErrorCount:     r.metrics.ErrorCount[name],
			CancelledCount: r.metrics.CancelledCount[name],
			TotalLatency:   r.metrics.TotalLatency[name],
			LastUsed:       r.metrics.LastUsed[name],
		}
	}
	return result
}

// ProviderMetrics holds metrics for a single provider
type ProviderMetrics struct {
	RequestCount   int64
	ErrorCount     int64
	CancelledCount int64
	TotalLatency   time.Duration
	LastUsed       time.Time
}

// Invoke implements the core.Runnable interface.
// It selects a provider using the routing strategy and invokes it.
func (r *Router) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	r.mu.RLock()
	if r.providers == nil {
		r.mu.RUnlock()
		return nil, ErrRouterClosed
	}
	r.mu.RUnlock()

	// Build request context for routing decision
	reqCtx := buildRequestContext(messages, opts)

	// Select provider using strategy
	startTime := time.Now()
	providerName, err := r.strategy.SelectProvider(ctx, reqCtx, r.getProvidersMap())
	if err != nil {
		return nil, NewRoutingError("strategy", err)
	}

	provider := r.GetProvider(providerName)
	if provider == nil {
		return nil, NewRoutingError("strategy", ErrProviderNotFound)
	}

	// Invoke selected provider
	response, err := provider.Invoke(ctx, messages, opts...)
	latency := time.Since(startTime)

	if err != nil {
		r.updateMetrics(providerName, latency, true)
		r.strategy.OnError(ctx, providerName, err)

		// Try fallback if configured
		if r.fallback != nil && r.fallback.ShouldRetry(err, 1) {
			return r.tryFallback(ctx, messages, opts, providerName, err)
		}

		return nil, err
	}

	r.updateMetrics(providerName, latency, false)
	r.strategy.OnSuccess(ctx, providerName, latency)

	return response, nil
}

// tryFallback attempts to use a fallback provider after the primary fails
func (r *Router) tryFallback(ctx context.Context, messages []core.Message, opts []core.Option, failedProvider string, originalErr error) (*core.AIMessage, error) {
	attemptedFallbacks := []string{failedProvider}
	currentProvider := failedProvider
	attemptCount := 1

	for {
		fallbackName, err := r.fallback.GetFallbackProvider(ctx, currentProvider, r.getProvidersMap())
		if err != nil {
			return nil, NewFallbackError(failedProvider, attemptedFallbacks, originalErr)
		}

		// Check if we've already tried this provider
		for _, attempted := range attemptedFallbacks {
			if attempted == fallbackName {
				return nil, NewFallbackError(failedProvider, attemptedFallbacks, originalErr)
			}
		}

		attemptedFallbacks = append(attemptedFallbacks, fallbackName)
		attemptCount++

		fallbackProvider := r.GetProvider(fallbackName)
		if fallbackProvider == nil {
			return nil, NewFallbackError(failedProvider, attemptedFallbacks, ErrProviderNotFound)
		}

		startTime := time.Now()
		response, err := fallbackProvider.Invoke(ctx, messages, opts...)
		latency := time.Since(startTime)

		if err != nil {
			r.updateMetrics(fallbackName, latency, true)
			r.strategy.OnError(ctx, fallbackName, err)

			if !r.fallback.ShouldRetry(err, attemptCount) {
				return nil, NewFallbackError(failedProvider, attemptedFallbacks, err)
			}

			currentProvider = fallbackName
			continue
		}

		r.updateMetrics(fallbackName, latency, false)
		r.strategy.OnSuccess(ctx, fallbackName, latency)
		return response, nil
	}
}

// getProvidersMap returns a copy of the providers map for routing decisions
func (r *Router) getProvidersMap() map[string]llms.ChatModel {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.providers == nil {
		return nil
	}

	result := make(map[string]llms.ChatModel, len(r.providers))
	for name, provider := range r.providers {
		result[name] = provider
	}
	return result
}

// buildRequestContext creates a RequestContext from messages and options
// Optimized to minimize allocations
func buildRequestContext(messages []core.Message, opts []core.Option) RequestContext {
	reqCtx := RequestContext{
		Messages:     messages,
		MessageCount: len(messages),
		TotalTokens:  0,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "moderate",
		UserMetadata: nil, // Lazy initialization only if needed
	}

	// Estimate token count and check for tool calls
	for _, msg := range messages {
		reqCtx.TotalTokens += estimateTokens(msg.GetContent())

		// Check for tool calls in AI messages
		if aiMsg, ok := msg.(*core.AIMessage); ok && len(aiMsg.ToolCalls) > 0 {
			reqCtx.HasToolCalls = true
		}
		// Check for tool messages (responses to tool calls)
		if _, ok := msg.(*core.ToolMessage); ok {
			reqCtx.HasToolCalls = true
		}
	}

	// Infer complexity from request characteristics
	if reqCtx.TotalTokens > 10000 || reqCtx.HasToolCalls {
		reqCtx.Complexity = "complex"
	} else if reqCtx.TotalTokens < 1000 && !reqCtx.HasToolCalls {
		reqCtx.Complexity = "simple"
	}

	return reqCtx
}

// estimateTokens provides a rough estimate of token count from content
func estimateTokens(content string) int {
	// Rough estimate: ~4 characters per token on average
	if len(content) < 4 {
		return 1
	}
	return len(content) / 4
}

// Stream implements the core.Runnable interface.
// It selects a provider using the routing strategy and streams from it.
func (r *Router) Stream(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	r.mu.RLock()
	if r.providers == nil {
		r.mu.RUnlock()
		return nil, ErrRouterClosed
	}
	r.mu.RUnlock()

	// Build request context for routing decision
	reqCtx := buildRequestContext(messages, opts)

	// Select provider using strategy
	providerName, err := r.strategy.SelectProvider(ctx, reqCtx, r.getProvidersMap())
	if err != nil {
		return nil, NewRoutingError("strategy", err)
	}

	provider := r.GetProvider(providerName)
	if provider == nil {
		return nil, NewRoutingError("strategy", ErrProviderNotFound)
	}

	// Stream from selected provider
	startTime := time.Now()
	iter, err := provider.Stream(ctx, messages, opts...)

	if err != nil {
		r.updateStreamMetrics(providerName, time.Since(startTime), true, false)
		r.strategy.OnError(ctx, providerName, err)
		return nil, err
	}

	outCh := make(chan core.StreamChunk[*core.AIMessage], 64)
	wrapped := core.NewStreamIterator(outCh)

	go func() {
		defer close(outCh)

		listenerDone := make(chan struct{})
		defer close(listenerDone)
		go func() {
			select {
			case <-wrapped.Done():
				iter.Close()
			case <-listenerDone:
			}
		}()

		cancelled := false
		var streamErr error
		defer func() {
			latency := time.Since(startTime)
			r.updateStreamMetrics(providerName, latency, streamErr != nil, cancelled)
			if streamErr != nil {
				r.strategy.OnError(ctx, providerName, streamErr)
				return
			}
			if !cancelled {
				r.strategy.OnSuccess(ctx, providerName, latency)
			}
		}()

		for {
			chunk, ok, err := iter.Next()
			if err != nil {
				streamErr = err
				outCh <- core.StreamChunk[*core.AIMessage]{Err: err}
				return
			}
			if !ok {
				select {
				case <-wrapped.Done():
					cancelled = true
				default:
				}
				return
			}

			select {
			case outCh <- core.StreamChunk[*core.AIMessage]{Value: chunk}:
			case <-wrapped.Done():
				cancelled = true
				iter.Close()
				return
			case <-ctx.Done():
				cancelled = true
				iter.Close()
				return
			}
		}
	}()

	return wrapped, nil
}

// Batch implements the core.Runnable interface.
// It processes multiple message sets in parallel using the routing strategy.
func (r *Router) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	r.mu.RLock()
	if r.providers == nil {
		r.mu.RUnlock()
		return nil, ErrRouterClosed
	}
	r.mu.RUnlock()

	results := make([]*core.AIMessage, len(inputs))
	errs := make([]error, len(inputs))

	cfg := core.ApplyOptions(opts...)
	limit := len(inputs)
	if cfg.MaxConcurrency > 0 && cfg.MaxConcurrency < limit {
		limit = cfg.MaxConcurrency
	}
	if limit <= 0 {
		limit = 1
	}
	sem := make(chan struct{}, limit)

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, msg []core.Message) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				errs[idx] = ctx.Err()
				return
			}
			defer func() { <-sem }()

			result, err := r.Invoke(ctx, msg, opts...)
			results[idx] = result
			errs[idx] = err
		}(i, input)
	}
	wg.Wait()

	failedItems := make(map[int]error)
	for idx, err := range errs {
		if err != nil {
			failedItems[idx] = err
		}
	}
	if len(failedItems) > 0 {
		return results, &BatchError{FailedItems: failedItems}
	}

	return results, nil
}

// GetName implements the core.Runnable interface.
func (r *Router) GetName() string {
	return "router"
}

// Generate implements the llms.ChatModel interface.
// It selects a provider using the routing strategy and generates a response.
func (r *Router) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	r.mu.RLock()
	if r.providers == nil {
		r.mu.RUnlock()
		return nil, ErrRouterClosed
	}
	r.mu.RUnlock()

	// Build request context for routing decision
	reqCtx := buildRequestContext(messages, opts)

	// Select provider using strategy
	startTime := time.Now()
	providerName, err := r.strategy.SelectProvider(ctx, reqCtx, r.getProvidersMap())
	if err != nil {
		return nil, NewRoutingError("strategy", err)
	}

	provider := r.GetProvider(providerName)
	if provider == nil {
		return nil, NewRoutingError("strategy", ErrProviderNotFound)
	}

	// Generate from selected provider
	result, err := provider.Generate(ctx, messages, opts...)
	latency := time.Since(startTime)

	if err != nil {
		r.updateMetrics(providerName, latency, true)
		r.strategy.OnError(ctx, providerName, err)

		// Try fallback if configured
		if r.fallback != nil && r.fallback.ShouldRetry(err, 1) {
			return r.tryFallbackGenerate(ctx, messages, opts, providerName, err)
		}

		return nil, err
	}

	r.updateMetrics(providerName, latency, false)
	r.strategy.OnSuccess(ctx, providerName, latency)

	return result, nil
}

// tryFallbackGenerate attempts to use a fallback provider for Generate
func (r *Router) tryFallbackGenerate(ctx context.Context, messages []core.Message, opts []core.Option, failedProvider string, originalErr error) (*llms.ChatResult, error) {
	attemptedFallbacks := []string{failedProvider}
	currentProvider := failedProvider
	attemptCount := 1

	for {
		fallbackName, err := r.fallback.GetFallbackProvider(ctx, currentProvider, r.getProvidersMap())
		if err != nil {
			return nil, NewFallbackError(failedProvider, attemptedFallbacks, originalErr)
		}

		// Check if we've already tried this provider
		for _, attempted := range attemptedFallbacks {
			if attempted == fallbackName {
				return nil, NewFallbackError(failedProvider, attemptedFallbacks, originalErr)
			}
		}

		attemptedFallbacks = append(attemptedFallbacks, fallbackName)
		attemptCount++

		fallbackProvider := r.GetProvider(fallbackName)
		if fallbackProvider == nil {
			return nil, NewFallbackError(failedProvider, attemptedFallbacks, ErrProviderNotFound)
		}

		startTime := time.Now()
		result, err := fallbackProvider.Generate(ctx, messages, opts...)
		latency := time.Since(startTime)

		if err != nil {
			r.updateMetrics(fallbackName, latency, true)
			r.strategy.OnError(ctx, fallbackName, err)

			if !r.fallback.ShouldRetry(err, attemptCount) {
				return nil, NewFallbackError(failedProvider, attemptedFallbacks, err)
			}

			currentProvider = fallbackName
			continue
		}

		r.updateMetrics(fallbackName, latency, false)
		r.strategy.OnSuccess(ctx, fallbackName, latency)
		return result, nil
	}
}

// BindTools implements the llms.ChatModel interface.
// It binds tools to all providers managed by the router.
func (r *Router) BindTools(tools ...llms.ToolDefinition) llms.ChatModel {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.providers == nil {
		return r
	}

	// Bind tools to all providers
	for name, provider := range r.providers {
		r.providers[name] = provider.BindTools(tools...)
	}

	return r
}

// WithStructuredOutput implements the llms.ChatModel interface.
// It configures structured output for all providers managed by the router.
func (r *Router) WithStructuredOutput(schema map[string]any) llms.ChatModel {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.providers == nil {
		return r
	}

	// Configure structured output for all providers
	for name, provider := range r.providers {
		r.providers[name] = provider.WithStructuredOutput(schema)
	}

	return r
}
