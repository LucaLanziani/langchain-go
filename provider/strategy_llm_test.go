package provider

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// mockLLMForRouting is a mock ChatModel for testing LLM routing strategy
type mockLLMForRouting struct {
	response     string
	err          error
	invocations  int
	mu           sync.Mutex
	shouldFail   bool
	failureCount int // Number of times to fail before succeeding
	invokeFunc   func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error)
}

func (m *mockLLMForRouting) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.invocations++

	// Use custom invoke function if provided
	if m.invokeFunc != nil {
		return m.invokeFunc(ctx, messages, opts...)
	}

	// Simulate failure for first N calls
	if m.shouldFail && m.invocations <= m.failureCount {
		return nil, m.err
	}

	return core.NewAIMessage(m.response), nil
}

func (m *mockLLMForRouting) Stream(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	ch := make(chan core.StreamChunk[*core.AIMessage], 1)
	ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(m.response)}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (m *mockLLMForRouting) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	for i := range inputs {
		results[i] = core.NewAIMessage(m.response)
	}
	return results, nil
}

func (m *mockLLMForRouting) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	msg, err := m.Invoke(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	return &llms.ChatResult{
		Generations: []*llms.ChatGeneration{{Message: msg}},
	}, nil
}

func (m *mockLLMForRouting) GetName() string {
	return "MockLLMForRouting"
}

func (m *mockLLMForRouting) BindTools(...llms.ToolDefinition) llms.ChatModel {
	return m
}

func (m *mockLLMForRouting) BindSkills(...llms.SkillDefinition) llms.ChatModel {
	return m
}

func (m *mockLLMForRouting) WithStructuredOutput(map[string]any) llms.ChatModel {
	return m
}

func (m *mockLLMForRouting) GetInvocations() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.invocations
}

// mockProviderForRouting is a simple mock provider for testing routing
type mockProviderForRouting struct {
	name string
}

func (m *mockProviderForRouting) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	return core.NewAIMessage(fmt.Sprintf("Response from %s", m.name)), nil
}

func (m *mockProviderForRouting) Stream(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.StreamIterator[*core.AIMessage], error) {
	ch := make(chan core.StreamChunk[*core.AIMessage], 1)
	ch <- core.StreamChunk[*core.AIMessage]{Value: core.NewAIMessage(fmt.Sprintf("Response from %s", m.name))}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

func (m *mockProviderForRouting) Batch(ctx context.Context, inputs [][]core.Message, opts ...core.Option) ([]*core.AIMessage, error) {
	results := make([]*core.AIMessage, len(inputs))
	for i := range inputs {
		results[i] = core.NewAIMessage(fmt.Sprintf("Response from %s", m.name))
	}
	return results, nil
}

func (m *mockProviderForRouting) Generate(ctx context.Context, messages []core.Message, opts ...core.Option) (*llms.ChatResult, error) {
	msg, err := m.Invoke(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	return &llms.ChatResult{
		Generations: []*llms.ChatGeneration{{Message: msg}},
	}, nil
}

func (m *mockProviderForRouting) GetName() string {
	return m.name
}

func (m *mockProviderForRouting) BindTools(...llms.ToolDefinition) llms.ChatModel {
	return m
}

func (m *mockProviderForRouting) BindSkills(...llms.SkillDefinition) llms.ChatModel {
	return m
}

func (m *mockProviderForRouting) WithStructuredOutput(map[string]any) llms.ChatModel {
	return m
}

// TestProperty21_LLMRoutingGracefulDegradation tests that LLM routing falls back
// gracefully when the LLM fails during SelectProvider().
//
// Property 21: LLM Routing Graceful Degradation
// ∀ router with LLMRoutingStrategy:
//
//	LLM call fails during SelectProvider()
//	⟹ returns valid provider (fallback) ∧ error is non-nil
//	∧ request still succeeds with fallback provider
//
// Validates: Requirement 11.4
func TestProperty21_LLMRoutingGracefulDegradation(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name        string
		llmError    error
		expectedErr bool
		description string
	}{
		{
			name:        "LLM network timeout",
			llmError:    errors.New("context deadline exceeded"),
			expectedErr: true,
			description: "LLM call times out, should fallback to available provider",
		},
		{
			name:        "LLM API error",
			llmError:    errors.New("API rate limit exceeded"),
			expectedErr: true,
			description: "LLM API fails, should fallback to available provider",
		},
		{
			name:        "LLM internal error",
			llmError:    errors.New("internal server error"),
			expectedErr: true,
			description: "LLM has internal error, should fallback to available provider",
		},
		{
			name:        "LLM authentication error",
			llmError:    errors.New("authentication failed"),
			expectedErr: true,
			description: "LLM auth fails, should fallback to available provider",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create mock LLM that fails
			mockLLM := &mockLLMForRouting{
				response:     "openai", // This won't be returned due to error
				err:          tc.llmError,
				shouldFail:   true,
				failureCount: 999, // Always fail
			}

			// Create available providers
			providers := map[string]llms.ChatModel{
				"openai":     &mockProviderForRouting{name: "openai"},
				"anthropic":  &mockProviderForRouting{name: "anthropic"},
				"localmodel": &mockProviderForRouting{name: "localmodel"},
			}

			// Create LLM routing strategy
			strategy := &LLMRoutingStrategy{
				model:     mockLLM,
				providers: []string{"openai", "anthropic", "localmodel"},
				providerDescriptions: map[string]string{
					"openai":     "Fast and cost-effective",
					"anthropic":  "Highly capable",
					"localmodel": "Local model",
				},
				cacheTTL: 5 * time.Minute,
			}

			// Create request context
			reqCtx := RequestContext{
				Messages: []core.Message{
					core.NewHumanMessage("Test message"),
				},
				MessageCount: 1,
				TotalTokens:  10,
				HasToolCalls: false,
				Priority:     "medium",
				Complexity:   "simple",
			}

			// Test 1: SelectProvider should return a valid provider despite LLM failure
			providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

			// Verify error is returned (indicating LLM failed)
			if tc.expectedErr && err == nil {
				t.Error("Expected error when LLM fails, got nil")
			}

			// Verify a valid provider is still returned (fallback)
			if providerName == "" {
				t.Fatal("Expected non-empty provider name as fallback, got empty string")
			}

			if _, exists := providers[providerName]; !exists {
				t.Errorf("Returned provider %s does not exist in available providers", providerName)
			}

			// Test 2: Verify LLM was actually called (not bypassed)
			if mockLLM.GetInvocations() == 0 {
				t.Error("Expected LLM to be invoked, but it was not called")
			}

			// Test 3: Verify the fallback provider can handle requests
			selectedProvider := providers[providerName]
			response, err := selectedProvider.Invoke(ctx, reqCtx.Messages)
			if err != nil {
				t.Errorf("Fallback provider failed to handle request: %v", err)
			}
			if response == nil {
				t.Error("Expected non-nil response from fallback provider")
			}

			t.Logf("✓ %s: LLM failed with '%v', gracefully fell back to provider '%s'",
				tc.description, tc.llmError, providerName)
		})
	}
}

// TestProperty21_LLMRoutingGracefulDegradation_InvalidProviderResponse tests that
// when the LLM returns an invalid provider name, the strategy falls back gracefully.
func TestProperty21_LLMRoutingGracefulDegradation_InvalidProviderResponse(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name        string
		llmResponse string
		description string
	}{
		{
			name:        "Non-existent provider",
			llmResponse: "nonexistent-provider",
			description: "LLM returns provider that doesn't exist",
		},
		{
			name:        "Empty response",
			llmResponse: "",
			description: "LLM returns empty string",
		},
		{
			name:        "Malformed response",
			llmResponse: "invalid response with spaces",
			description: "LLM returns malformed provider name",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create mock LLM that returns invalid provider
			mockLLM := &mockLLMForRouting{
				response:   tc.llmResponse,
				err:        nil,
				shouldFail: false,
			}

			// Create available providers
			providers := map[string]llms.ChatModel{
				"openai":    &mockProviderForRouting{name: "openai"},
				"anthropic": &mockProviderForRouting{name: "anthropic"},
			}

			// Create LLM routing strategy
			strategy := &LLMRoutingStrategy{
				model:     mockLLM,
				providers: []string{"openai", "anthropic"},
				providerDescriptions: map[string]string{
					"openai":    "Fast and cost-effective",
					"anthropic": "Highly capable",
				},
				cacheTTL: 5 * time.Minute,
			}

			// Create request context
			reqCtx := RequestContext{
				Messages: []core.Message{
					core.NewHumanMessage("Test message"),
				},
				MessageCount: 1,
				TotalTokens:  10,
				HasToolCalls: false,
				Priority:     "medium",
				Complexity:   "simple",
			}

			// SelectProvider should return a valid provider despite invalid LLM response
			providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

			// Verify error is returned (indicating invalid provider from LLM)
			if err == nil {
				t.Error("Expected error when LLM returns invalid provider, got nil")
			}

			// Verify a valid provider is still returned (fallback)
			if providerName == "" {
				t.Fatal("Expected non-empty provider name as fallback, got empty string")
			}

			if _, exists := providers[providerName]; !exists {
				t.Errorf("Returned provider %s does not exist in available providers", providerName)
			}

			// Verify the fallback provider can handle requests
			selectedProvider := providers[providerName]
			response, err := selectedProvider.Invoke(ctx, reqCtx.Messages)
			if err != nil {
				t.Errorf("Fallback provider failed to handle request: %v", err)
			}
			if response == nil {
				t.Error("Expected non-nil response from fallback provider")
			}

			t.Logf("✓ %s: LLM returned invalid '%s', gracefully fell back to provider '%s'",
				tc.description, tc.llmResponse, providerName)
		})
	}
}

// TestProperty21_LLMRoutingGracefulDegradation_NilLLM tests that when no LLM is
// configured, the strategy falls back gracefully.
func TestProperty21_LLMRoutingGracefulDegradation_NilLLM(t *testing.T) {
	ctx := context.Background()

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy with nil model
	strategy := &LLMRoutingStrategy{
		model:     nil, // No LLM configured
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  10,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// SelectProvider should return a valid provider despite nil LLM
	providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

	// Verify error is returned (indicating LLM not configured)
	if err == nil {
		t.Error("Expected error when LLM is nil, got nil")
	}

	// Verify a valid provider is still returned (fallback)
	if providerName == "" {
		t.Fatal("Expected non-empty provider name as fallback, got empty string")
	}

	if _, exists := providers[providerName]; !exists {
		t.Errorf("Returned provider %s does not exist in available providers", providerName)
	}

	// Verify the fallback provider can handle requests
	selectedProvider := providers[providerName]
	response, err := selectedProvider.Invoke(ctx, reqCtx.Messages)
	if err != nil {
		t.Errorf("Fallback provider failed to handle request: %v", err)
	}
	if response == nil {
		t.Error("Expected non-nil response from fallback provider")
	}

	t.Logf("✓ LLM not configured (nil), gracefully fell back to provider '%s'", providerName)
}

// TestProperty21_LLMRoutingGracefulDegradation_ConcurrentFailures tests that
// graceful degradation works correctly under concurrent load.
func TestProperty21_LLMRoutingGracefulDegradation_ConcurrentFailures(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM that fails
	mockLLM := &mockLLMForRouting{
		response:     "openai",
		err:          errors.New("concurrent failure test"),
		shouldFail:   true,
		failureCount: 999,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":     &mockProviderForRouting{name: "openai"},
		"anthropic":  &mockProviderForRouting{name: "anthropic"},
		"localmodel": &mockProviderForRouting{name: "localmodel"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic", "localmodel"},
		providerDescriptions: map[string]string{
			"openai":     "Fast and cost-effective",
			"anthropic":  "Highly capable",
			"localmodel": "Local model",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  10,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// Run concurrent requests
	concurrency := 50
	var wg sync.WaitGroup
	errorsChan := make(chan error, concurrency)
	providerNames := make(chan string, concurrency)

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

			if err != nil {
				errorsChan <- err
			}

			if providerName != "" {
				providerNames <- providerName
			}
		}()
	}

	wg.Wait()
	close(errorsChan)
	close(providerNames)

	// Verify all requests got errors (LLM failed)
	errorCount := 0
	for range errorsChan {
		errorCount++
	}
	if errorCount != concurrency {
		t.Errorf("Expected %d errors, got %d", concurrency, errorCount)
	}

	// Verify all requests got valid fallback providers
	providerCount := 0
	for providerName := range providerNames {
		providerCount++
		if _, exists := providers[providerName]; !exists {
			t.Errorf("Invalid provider returned: %s", providerName)
		}
	}
	if providerCount != concurrency {
		t.Errorf("Expected %d provider names, got %d", concurrency, providerCount)
	}

	t.Logf("✓ All %d concurrent requests gracefully fell back despite LLM failures", concurrency)
}

// TestProperty21_LLMRoutingGracefulDegradation_RecoveryAfterFailure tests that
// the strategy can recover and use the LLM again after it starts working.
func TestProperty21_LLMRoutingGracefulDegradation_RecoveryAfterFailure(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM that fails first 3 times, then succeeds
	mockLLM := &mockLLMForRouting{
		response:     "anthropic",
		err:          errors.New("temporary failure"),
		shouldFail:   true,
		failureCount: 3,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  10,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// First 3 calls should fail and fallback
	for i := 1; i <= 3; i++ {
		providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

		if err == nil {
			t.Errorf("Call %d: Expected error when LLM fails, got nil", i)
		}

		if providerName == "" {
			t.Errorf("Call %d: Expected fallback provider, got empty string", i)
		}

		if _, exists := providers[providerName]; !exists {
			t.Errorf("Call %d: Invalid fallback provider: %s", i, providerName)
		}

		t.Logf("Call %d: LLM failed, fell back to '%s'", i, providerName)
	}

	// 4th call should succeed (LLM recovered)
	providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)

	if err != nil {
		t.Errorf("Call 4: Expected success after LLM recovery, got error: %v", err)
	}

	if providerName != "anthropic" {
		t.Errorf("Call 4: Expected 'anthropic' from LLM, got '%s'", providerName)
	}

	t.Logf("✓ Call 4: LLM recovered, successfully routed to '%s'", providerName)
}

// TestProperty22_LLMRoutingCacheConsistency tests that the cache returns valid
// providers and respects TTL.
//
// Property 22: LLM Routing Cache Consistency
// ∀ router with LLMRoutingStrategy, cacheKey:
//
//	cachedProvider := cache[cacheKey]
//	⟹ (cachedProvider ∈ availableProviders) ∨ (cache miss)
//	∧ cache entries expire after cacheTTL
//
// Validates: Requirements 11.5, 11.6, 11.7, 23.3
func TestProperty22_LLMRoutingCacheConsistency(t *testing.T) {
	ctx := context.Background()

	t.Run("cached provider is always valid", func(t *testing.T) {
		// Create mock LLM that returns a valid provider
		mockLLM := &mockLLMForRouting{
			response:   "anthropic",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy with cache
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  10,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "simple",
		}

		// First call - should call LLM and cache result
		providerName1, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		// Verify provider is valid
		if _, exists := providers[providerName1]; !exists {
			t.Errorf("First call returned invalid provider: %s", providerName1)
		}

		// Second call with same request - should use cache
		providerName2, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("Second SelectProvider call failed: %v", err)
		}

		// Verify cached provider is still valid
		if _, exists := providers[providerName2]; !exists {
			t.Errorf("Cached provider is invalid: %s", providerName2)
		}

		// Verify cache was used (same provider returned)
		if providerName1 != providerName2 {
			t.Errorf("Cache not used: first=%s, second=%s", providerName1, providerName2)
		}

		// Verify LLM was only called once (cache hit on second call)
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation (cache hit), got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Cached provider '%s' is valid and cache was used", providerName2)
	})

	t.Run("cache entries expire after TTL", func(t *testing.T) {
		// Create mock LLM
		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
		}

		// Create LLM routing strategy with SHORT cache TTL
		shortTTL := 100 * time.Millisecond
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic"},
			providerDescriptions: map[string]string{
				"openai":    "Fast and cost-effective",
				"anthropic": "Highly capable",
			},
			cacheTTL: shortTTL,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  10,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "simple",
		}

		// First call - should call LLM and cache result
		_, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		// Verify LLM was called once
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
		}

		// Second call immediately - should use cache
		_, err = strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("Second SelectProvider call failed: %v", err)
		}

		// Verify LLM was NOT called again (cache hit)
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation (cache hit), got %d", mockLLM.GetInvocations())
		}

		// Wait for cache to expire
		time.Sleep(shortTTL + 50*time.Millisecond)

		// Third call after TTL - should call LLM again (cache expired)
		_, err = strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("Third SelectProvider call failed: %v", err)
		}

		// Verify LLM was called again (cache expired)
		if mockLLM.GetInvocations() != 2 {
			t.Errorf("Expected 2 LLM invocations (cache expired), got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Cache expired after %v and LLM was called again", shortTTL)
	})

	t.Run("cache miss when provider no longer available", func(t *testing.T) {
		// Create mock LLM that returns "copilot"
		mockLLM := &mockLLMForRouting{
			response:   "copilot",
			err:        nil,
			shouldFail: false,
		}

		// Create initial providers including copilot
		initialProviders := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
			"copilot":   &mockProviderForRouting{name: "copilot"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "copilot"},
			providerDescriptions: map[string]string{
				"openai":    "Fast and cost-effective",
				"anthropic": "Highly capable",
				"copilot":   "Assistant-style provider",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  10,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "simple",
		}

		// First call - should cache "copilot"
		providerName1, err := strategy.SelectProvider(ctx, reqCtx, initialProviders)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		if providerName1 != "copilot" {
			t.Errorf("Expected 'copilot', got '%s'", providerName1)
		}

		// Simulate copilot becoming unavailable
		providersWithoutCopilot := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
		}

		// Second call - cached provider "copilot" is no longer available
		// Should detect this and call LLM again, but LLM still returns "copilot"
		// Strategy should fallback to an available provider
		providerName2, err := strategy.SelectProvider(ctx, reqCtx, providersWithoutCopilot)

		// Error is expected because LLM returned invalid provider
		if err == nil {
			t.Error("Expected error when LLM returns unavailable provider, got nil")
		}

		// But a valid fallback provider should still be returned
		if providerName2 == "" {
			t.Fatal("Expected fallback provider, got empty string")
		}

		// Verify returned provider is valid (not the cached "copilot")
		if _, exists := providersWithoutCopilot[providerName2]; !exists {
			t.Errorf("Returned provider '%s' is not in available providers", providerName2)
		}

		// Verify it's not the cached provider
		if providerName2 == "copilot" {
			t.Error("Should not return cached provider 'copilot' when it's unavailable")
		}

		t.Logf("✓ Cache miss detected when cached provider 'copilot' became unavailable, fell back to '%s'", providerName2)
	})

	t.Run("cache consistency under concurrent access", func(t *testing.T) {
		// Create mock LLM
		mockLLM := &mockLLMForRouting{
			response:   "anthropic",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  10,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "simple",
		}

		// Run concurrent requests with same request context
		concurrency := 50
		var wg sync.WaitGroup
		providerNames := make(chan string, concurrency)
		errors := make(chan error, concurrency)

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)
				if err != nil {
					errors <- err
					return
				}

				// Verify provider is valid
				if _, exists := providers[providerName]; !exists {
					errors <- fmt.Errorf("invalid provider: %s", providerName)
					return
				}

				providerNames <- providerName
			}()
		}

		wg.Wait()
		close(providerNames)
		close(errors)

		// Check for errors
		for err := range errors {
			t.Errorf("Concurrent request failed: %v", err)
		}

		// Verify all requests got valid providers
		providerCount := 0
		firstProvider := ""
		allSame := true

		for providerName := range providerNames {
			providerCount++
			if firstProvider == "" {
				firstProvider = providerName
			} else if providerName != firstProvider {
				allSame = false
			}
		}

		if providerCount != concurrency {
			t.Errorf("Expected %d provider names, got %d", concurrency, providerCount)
		}

		// All concurrent requests with same context should get same cached provider
		if !allSame {
			t.Error("Expected all concurrent requests to get same cached provider")
		}

		// LLM should be called only once (all others hit cache)
		invocations := mockLLM.GetInvocations()
		if invocations > 5 {
			t.Errorf("Expected few LLM invocations due to caching, got %d", invocations)
		}

		t.Logf("✓ %d concurrent requests handled consistently, LLM called %d times", concurrency, invocations)
	})

	t.Run("different request contexts produce different cache keys", func(t *testing.T) {
		// Create mock LLM
		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic"},
			providerDescriptions: map[string]string{
				"openai":    "Fast and cost-effective",
				"anthropic": "Highly capable",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create two different request contexts
		reqCtx1 := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Simple message"),
			},
			MessageCount: 1,
			TotalTokens:  10,
			HasToolCalls: false,
			Priority:     "low",
			Complexity:   "simple",
		}

		reqCtx2 := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Complex message with tools"),
			},
			MessageCount: 1,
			TotalTokens:  5000,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		// Call with first context
		provider1, err := strategy.SelectProvider(ctx, reqCtx1, providers)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		// Call with second context (different cache key)
		provider2, err := strategy.SelectProvider(ctx, reqCtx2, providers)
		if err != nil {
			t.Fatalf("Second SelectProvider call failed: %v", err)
		}

		// Verify both providers are valid
		if _, exists := providers[provider1]; !exists {
			t.Errorf("First provider invalid: %s", provider1)
		}
		if _, exists := providers[provider2]; !exists {
			t.Errorf("Second provider invalid: %s", provider2)
		}

		// Verify LLM was called twice (different cache keys)
		invocations := mockLLM.GetInvocations()
		if invocations != 2 {
			t.Errorf("Expected 2 LLM invocations (different cache keys), got %d", invocations)
		}

		t.Logf("✓ Different contexts produced separate cache entries, LLM called %d times", invocations)
	})

	t.Run("cache key generation is deterministic", func(t *testing.T) {
		strategy := &LLMRoutingStrategy{
			cacheTTL: 5 * time.Minute,
		}

		// Create identical request contexts
		reqCtx1 := RequestContext{
			MessageCount: 5,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		reqCtx2 := RequestContext{
			MessageCount: 5,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		// Generate cache keys
		key1 := strategy.generateCacheKey(reqCtx1)
		key2 := strategy.generateCacheKey(reqCtx2)

		// Verify keys are identical
		if key1 != key2 {
			t.Errorf("Expected identical cache keys for identical contexts, got '%s' vs '%s'", key1, key2)
		}

		// Verify keys are non-empty
		if key1 == "" {
			t.Error("Cache key should not be empty")
		}

		t.Logf("✓ Identical contexts produce identical cache key: %s", key1)
	})
}

// TestProperty23_LLMRoutingDeterminismWithCache tests that when the cache is hit,
// the same provider is always returned for identical requests.
//
// Property 23: LLM Routing Determinism with Cache
// ∀ router with LLMRoutingStrategy, identical requests R1, R2:
//
//	cache hit for R1 ⟹ SelectProvider(R1) = SelectProvider(R2)
//	∧ no LLM call for R2 (cache hit)
//	∧ provider returned is valid
//
// Validates: Requirement 23.2
func TestProperty23_LLMRoutingDeterminismWithCache(t *testing.T) {
	ctx := context.Background()

	t.Run("identical requests return same cached provider", func(t *testing.T) {
		// Create mock LLM that returns "anthropic"
		mockLLM := &mockLLMForRouting{
			response:   "anthropic",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy with cache
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create identical request contexts
		reqCtx1 := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		reqCtx2 := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		// First request - should call LLM and cache result
		provider1, err := strategy.SelectProvider(ctx, reqCtx1, providers)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		// Verify provider is valid
		if _, exists := providers[provider1]; !exists {
			t.Errorf("First provider invalid: %s", provider1)
		}

		// Verify LLM was called once
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
		}

		// Second request with identical context - should use cache
		provider2, err := strategy.SelectProvider(ctx, reqCtx2, providers)
		if err != nil {
			t.Fatalf("Second SelectProvider call failed: %v", err)
		}

		// Verify provider is valid
		if _, exists := providers[provider2]; !exists {
			t.Errorf("Second provider invalid: %s", provider2)
		}

		// Property: Identical requests return same provider (determinism)
		if provider1 != provider2 {
			t.Errorf("Determinism violated: first=%s, second=%s", provider1, provider2)
		}

		// Property: No LLM call for second request (cache hit)
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation (cache hit), got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Identical requests deterministically returned '%s' (cache hit, no LLM call)", provider1)
	})

	t.Run("determinism holds across multiple identical requests", func(t *testing.T) {
		// Create mock LLM
		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic"},
			providerDescriptions: map[string]string{
				"openai":    "Fast and cost-effective",
				"anthropic": "Highly capable",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  2000,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "moderate",
		}

		// Make 10 identical requests
		numRequests := 10
		providerResults := make([]string, numRequests)

		for i := 0; i < numRequests; i++ {
			provider, err := strategy.SelectProvider(ctx, reqCtx, providers)
			if err != nil {
				t.Fatalf("Request %d failed: %v", i+1, err)
			}

			providerResults[i] = provider

			// Verify provider is valid
			if _, exists := providers[provider]; !exists {
				t.Errorf("Request %d returned invalid provider: %s", i+1, provider)
			}
		}

		// Property: All requests return the same provider (determinism)
		firstProvider := providerResults[0]
		for i, provider := range providerResults {
			if provider != firstProvider {
				t.Errorf("Determinism violated at request %d: expected=%s, got=%s", i+1, firstProvider, provider)
			}
		}

		// Property: LLM called only once (all others hit cache)
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ %d identical requests deterministically returned '%s' (1 LLM call, %d cache hits)",
			numRequests, firstProvider, numRequests-1)
	})

	t.Run("determinism under concurrent identical requests", func(t *testing.T) {
		// Create mock LLM
		mockLLM := &mockLLMForRouting{
			response:   "anthropic",
			err:        nil,
			shouldFail: false,
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create identical request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Concurrent test message"),
			},
			MessageCount: 1,
			TotalTokens:  3000,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		// Run 100 concurrent identical requests
		concurrency := 100
		var wg sync.WaitGroup
		providerNames := make(chan string, concurrency)
		errors := make(chan error, concurrency)

		for i := 0; i < concurrency; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()

				provider, err := strategy.SelectProvider(ctx, reqCtx, providers)
				if err != nil {
					errors <- err
					return
				}

				// Verify provider is valid
				if _, exists := providers[provider]; !exists {
					errors <- fmt.Errorf("invalid provider: %s", provider)
					return
				}

				providerNames <- provider
			}()
		}

		wg.Wait()
		close(providerNames)
		close(errors)

		// Check for errors
		for err := range errors {
			t.Errorf("Concurrent request failed: %v", err)
		}

		// Property: All concurrent requests return the same provider (determinism)
		providerSet := make(map[string]int)
		for provider := range providerNames {
			providerSet[provider]++
		}

		if len(providerSet) != 1 {
			t.Errorf("Determinism violated: got %d different providers: %v", len(providerSet), providerSet)
		}

		// Get the single provider name
		var singleProvider string
		for provider := range providerSet {
			singleProvider = provider
		}

		// Property: LLM called minimal times (most requests hit cache)
		invocations := mockLLM.GetInvocations()
		if invocations > 10 {
			t.Errorf("Expected few LLM invocations due to caching, got %d", invocations)
		}

		t.Logf("✓ %d concurrent identical requests deterministically returned '%s' (%d LLM calls, %d cache hits)",
			concurrency, singleProvider, invocations, concurrency-invocations)
	})

	t.Run("determinism breaks after cache expiration", func(t *testing.T) {
		// Create mock LLM that returns different providers on each call
		callCount := 0
		var mu sync.Mutex

		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		// Set custom invoke function that returns different providers
		mockLLM.invokeFunc = func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
			mu.Lock()
			callCount++
			currentCall := callCount
			mu.Unlock()

			// Return different provider on each call
			if currentCall == 1 {
				return core.NewAIMessage("openai"), nil
			}
			return core.NewAIMessage("anthropic"), nil
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":    &mockProviderForRouting{name: "openai"},
			"anthropic": &mockProviderForRouting{name: "anthropic"},
		}

		// Create LLM routing strategy with SHORT cache TTL
		shortTTL := 100 * time.Millisecond
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic"},
			providerDescriptions: map[string]string{
				"openai":    "Fast and cost-effective",
				"anthropic": "Highly capable",
			},
			cacheTTL: shortTTL,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  1000,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "simple",
		}

		// First request - should cache "openai"
		provider1, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("First SelectProvider call failed: %v", err)
		}

		if provider1 != "openai" {
			t.Errorf("Expected 'openai' on first call, got '%s'", provider1)
		}

		// Second request immediately - should return cached "openai"
		provider2, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("Second SelectProvider call failed: %v", err)
		}

		// Property: Determinism holds while cache is valid
		if provider2 != provider1 {
			t.Errorf("Determinism violated before expiration: first=%s, second=%s", provider1, provider2)
		}

		// Wait for cache to expire
		time.Sleep(shortTTL + 50*time.Millisecond)

		// Third request after expiration - should call LLM again and get "anthropic"
		provider3, err := strategy.SelectProvider(ctx, reqCtx, providers)
		if err != nil {
			t.Fatalf("Third SelectProvider call failed: %v", err)
		}

		if provider3 != "anthropic" {
			t.Errorf("Expected 'anthropic' after cache expiration, got '%s'", provider3)
		}

		// Property: Determinism can change after cache expiration (LLM called again)
		if provider3 == provider1 {
			t.Logf("Note: Provider remained same after expiration (LLM returned same provider)")
		}

		// Verify LLM was called twice (initial + after expiration)
		if mockLLM.GetInvocations() != 2 {
			t.Errorf("Expected 2 LLM invocations, got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Determinism held during cache validity ('%s'), then LLM called again after expiration (got '%s')",
			provider1, provider3)
	})

	t.Run("different request contexts have independent determinism", func(t *testing.T) {
		// Create mock LLM that returns different providers based on complexity
		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		// Override Invoke to return different providers based on request
		mockLLM.invokeFunc = func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
			// Parse the prompt to determine complexity
			if len(messages) > 0 {
				// Get the last message which contains the routing prompt
				lastMsg := messages[len(messages)-1]
				// Use GetContent() method to access content
				content := lastMsg.GetContent()
				if strings.Contains(content, "Complexity: simple") {
					return core.NewAIMessage("openai"), nil
				} else if strings.Contains(content, "Complexity: complex") {
					return core.NewAIMessage("anthropic"), nil
				}
			}

			return core.NewAIMessage("localmodel"), nil
		}

		// Create available providers
		providers := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create two different request contexts
		simpleReqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Simple message"),
			},
			MessageCount: 1,
			TotalTokens:  500,
			HasToolCalls: false,
			Priority:     "low",
			Complexity:   "simple",
		}

		complexReqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Complex message"),
			},
			MessageCount: 1,
			TotalTokens:  5000,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		// Make requests with simple context (should cache "openai")
		simpleProvider1, err := strategy.SelectProvider(ctx, simpleReqCtx, providers)
		if err != nil {
			t.Fatalf("First simple request failed: %v", err)
		}

		simpleProvider2, err := strategy.SelectProvider(ctx, simpleReqCtx, providers)
		if err != nil {
			t.Fatalf("Second simple request failed: %v", err)
		}

		// Property: Determinism for simple requests
		if simpleProvider1 != simpleProvider2 {
			t.Errorf("Simple request determinism violated: first=%s, second=%s", simpleProvider1, simpleProvider2)
		}

		// Make requests with complex context (should cache "anthropic")
		complexProvider1, err := strategy.SelectProvider(ctx, complexReqCtx, providers)
		if err != nil {
			t.Fatalf("First complex request failed: %v", err)
		}

		complexProvider2, err := strategy.SelectProvider(ctx, complexReqCtx, providers)
		if err != nil {
			t.Fatalf("Second complex request failed: %v", err)
		}

		// Property: Determinism for complex requests
		if complexProvider1 != complexProvider2 {
			t.Errorf("Complex request determinism violated: first=%s, second=%s", complexProvider1, complexProvider2)
		}

		// Property: Different contexts can have different cached providers
		// (This is expected - different cache keys)
		if simpleProvider1 == complexProvider1 {
			t.Logf("Note: Both contexts routed to same provider '%s' (valid but not required)", simpleProvider1)
		}

		// Verify LLM was called twice (once per unique context)
		if mockLLM.GetInvocations() != 2 {
			t.Errorf("Expected 2 LLM invocations (one per context), got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Independent determinism: simple requests → '%s', complex requests → '%s'",
			simpleProvider1, complexProvider1)
	})

	t.Run("cache provides determinism even when LLM is non-deterministic", func(t *testing.T) {
		// Create mock LLM that returns random providers (non-deterministic)
		providerList := []string{"openai", "anthropic", "localmodel"}
		callCount := 0
		var mu sync.Mutex

		mockLLM := &mockLLMForRouting{
			response:   "openai",
			err:        nil,
			shouldFail: false,
		}

		mockLLM.invokeFunc = func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
			mu.Lock()
			callCount++
			currentCall := callCount
			mu.Unlock()

			// Return different provider on each call (simulating non-deterministic LLM)
			provider := providerList[currentCall%len(providerList)]
			return core.NewAIMessage(provider), nil
		}

		// Create available providers
		availableProviders := map[string]llms.ChatModel{
			"openai":     &mockProviderForRouting{name: "openai"},
			"anthropic":  &mockProviderForRouting{name: "anthropic"},
			"localmodel": &mockProviderForRouting{name: "localmodel"},
		}

		// Create LLM routing strategy
		strategy := &LLMRoutingStrategy{
			model:     mockLLM,
			providers: []string{"openai", "anthropic", "localmodel"},
			providerDescriptions: map[string]string{
				"openai":     "Fast and cost-effective",
				"anthropic":  "Highly capable",
				"localmodel": "Local model",
			},
			cacheTTL: 5 * time.Minute,
		}

		// Create request context
		reqCtx := RequestContext{
			Messages: []core.Message{
				core.NewHumanMessage("Test message"),
			},
			MessageCount: 1,
			TotalTokens:  1000,
			HasToolCalls: false,
			Priority:     "medium",
			Complexity:   "moderate",
		}

		// First request - LLM returns some provider
		firstProvider, err := strategy.SelectProvider(ctx, reqCtx, availableProviders)
		if err != nil {
			t.Fatalf("First request failed: %v", err)
		}

		// Make 20 more identical requests
		for i := 0; i < 20; i++ {
			provider, err := strategy.SelectProvider(ctx, reqCtx, availableProviders)
			if err != nil {
				t.Fatalf("Request %d failed: %v", i+2, err)
			}

			// Property: Cache provides determinism even with non-deterministic LLM
			if provider != firstProvider {
				t.Errorf("Determinism violated at request %d: expected=%s, got=%s", i+2, firstProvider, provider)
			}
		}

		// Verify LLM was called only once (cache provided determinism)
		if mockLLM.GetInvocations() != 1 {
			t.Errorf("Expected 1 LLM invocation (cache hit for rest), got %d", mockLLM.GetInvocations())
		}

		t.Logf("✓ Cache provided determinism ('%s') despite non-deterministic LLM (1 call, 20 cache hits)", firstProvider)
	})
}

func TestLLMRoutingStrategyCoalescesConcurrentCacheMisses(t *testing.T) {
	ctx := context.Background()
	mockLLM := &mockLLMForRouting{
		invokeFunc: func(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
			time.Sleep(25 * time.Millisecond)
			return core.NewAIMessage("anthropic"), nil
		},
	}
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast",
			"anthropic": "Capable",
		},
		cacheTTL: 5 * time.Minute,
	}
	reqCtx := RequestContext{
		Messages:     []core.Message{core.NewHumanMessage("route me")},
		MessageCount: 1,
		TotalTokens:  10,
		Priority:     "medium",
		Complexity:   "simple",
	}

	const concurrency = 50
	var wg sync.WaitGroup
	results := make(chan string, concurrency)
	errs := make(chan error, concurrency)
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			providerName, err := strategy.SelectProvider(ctx, reqCtx, providers)
			if err != nil {
				errs <- err
				return
			}
			results <- providerName
		}()
	}
	wg.Wait()
	close(results)
	close(errs)

	for err := range errs {
		t.Fatalf("unexpected coalesced routing error: %v", err)
	}
	for providerName := range results {
		if providerName != "anthropic" {
			t.Fatalf("expected all coalesced requests to route to anthropic, got %q", providerName)
		}
	}
	if invocations := mockLLM.GetInvocations(); invocations != 1 {
		t.Fatalf("expected exactly 1 LLM invocation, got %d", invocations)
	}
}

// ===== Unit Tests for LLM Routing Strategy =====
// These tests validate specific scenarios and edge cases for Requirements 11.1-11.7, 23.1-23.5

// TestLLMRouting_CacheHit tests that when a cache entry exists and is valid,
// the cached provider is returned without calling the LLM.
// Validates: Requirements 11.5, 11.6, 23.1, 23.2
func TestLLMRouting_CacheHit(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM
	mockLLM := &mockLLMForRouting{
		response:   "anthropic",
		err:        nil,
		shouldFail: false,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  1000,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// First call - should call LLM and cache result
	provider1, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("First SelectProvider call failed: %v", err)
	}

	if provider1 != "anthropic" {
		t.Errorf("Expected 'anthropic', got '%s'", provider1)
	}

	// Verify LLM was called once
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
	}

	// Second call with same context - should use cache (no LLM call)
	provider2, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("Second SelectProvider call failed: %v", err)
	}

	if provider2 != "anthropic" {
		t.Errorf("Expected cached 'anthropic', got '%s'", provider2)
	}

	// Verify LLM was NOT called again (cache hit)
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation (cache hit), got %d", mockLLM.GetInvocations())
	}

	t.Log("✓ Cache hit returned cached provider without calling LLM")
}

// TestLLMRouting_CacheMiss tests that when no cache entry exists,
// the LLM is called to make a routing decision.
// Validates: Requirements 11.1, 11.2, 23.1
func TestLLMRouting_CacheMiss(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM
	mockLLM := &mockLLMForRouting{
		response:   "openai",
		err:        nil,
		shouldFail: false,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  2000,
		HasToolCalls: false,
		Priority:     "high",
		Complexity:   "moderate",
	}

	// Call SelectProvider - should call LLM (cache miss)
	provider, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("SelectProvider call failed: %v", err)
	}

	if provider != "openai" {
		t.Errorf("Expected 'openai' from LLM, got '%s'", provider)
	}

	// Verify LLM was called
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation (cache miss), got %d", mockLLM.GetInvocations())
	}

	t.Log("✓ Cache miss triggered LLM call and returned provider")
}

// TestLLMRouting_LLMFailureFallback tests that when the LLM call fails,
// the strategy falls back gracefully to an available provider.
// Validates: Requirement 11.4
func TestLLMRouting_LLMFailureFallback(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM that fails
	mockLLM := &mockLLMForRouting{
		response:     "openai",
		err:          errors.New("LLM API error"),
		shouldFail:   true,
		failureCount: 999,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  1500,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// Call SelectProvider - LLM fails, should fallback
	provider, err := strategy.SelectProvider(ctx, reqCtx, providers)

	// Error should be returned (indicating LLM failed)
	if err == nil {
		t.Error("Expected error when LLM fails, got nil")
	}

	// But a valid fallback provider should still be returned
	if provider == "" {
		t.Fatal("Expected fallback provider, got empty string")
	}

	if _, exists := providers[provider]; !exists {
		t.Errorf("Fallback provider '%s' does not exist in available providers", provider)
	}

	// Verify LLM was called (not bypassed)
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
	}

	t.Logf("✓ LLM failure fell back gracefully to provider '%s'", provider)
}

// TestLLMRouting_InvalidProviderFromLLM tests that when the LLM returns
// an invalid provider name, the strategy handles it correctly.
// Validates: Requirement 11.3
func TestLLMRouting_InvalidProviderFromLLM(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name        string
		llmResponse string
	}{
		{
			name:        "non-existent provider",
			llmResponse: "nonexistent-provider",
		},
		{
			name:        "empty response",
			llmResponse: "",
		},
		{
			name:        "malformed response",
			llmResponse: "this is not a valid provider name",
		},
		{
			name:        "provider with extra text",
			llmResponse: "openai is the best choice for this request",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create mock LLM that returns invalid provider
			mockLLM := &mockLLMForRouting{
				response:   tc.llmResponse,
				err:        nil,
				shouldFail: false,
			}

			// Create available providers
			providers := map[string]llms.ChatModel{
				"openai":    &mockProviderForRouting{name: "openai"},
				"anthropic": &mockProviderForRouting{name: "anthropic"},
			}

			// Create LLM routing strategy
			strategy := &LLMRoutingStrategy{
				model:     mockLLM,
				providers: []string{"openai", "anthropic"},
				providerDescriptions: map[string]string{
					"openai":    "Fast and cost-effective",
					"anthropic": "Highly capable",
				},
				cacheTTL: 5 * time.Minute,
			}

			// Create request context
			reqCtx := RequestContext{
				Messages: []core.Message{
					core.NewHumanMessage("Test message"),
				},
				MessageCount: 1,
				TotalTokens:  1000,
				HasToolCalls: false,
				Priority:     "medium",
				Complexity:   "simple",
			}

			// Call SelectProvider - LLM returns invalid provider
			provider, err := strategy.SelectProvider(ctx, reqCtx, providers)

			// Error should be returned (indicating invalid provider)
			if err == nil {
				t.Error("Expected error when LLM returns invalid provider, got nil")
			}

			// But a valid fallback provider should still be returned
			if provider == "" {
				t.Fatal("Expected fallback provider, got empty string")
			}

			if _, exists := providers[provider]; !exists {
				t.Errorf("Fallback provider '%s' does not exist in available providers", provider)
			}

			t.Logf("✓ Invalid LLM response '%s' handled, fell back to '%s'", tc.llmResponse, provider)
		})
	}
}

// TestLLMRouting_CacheExpiration tests that cache entries expire after TTL
// and the LLM is called again for expired entries.
// Validates: Requirements 11.7, 23.3
func TestLLMRouting_CacheExpiration(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM
	mockLLM := &mockLLMForRouting{
		response:   "anthropic",
		err:        nil,
		shouldFail: false,
	}

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy with SHORT cache TTL
	shortTTL := 100 * time.Millisecond
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: shortTTL,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  1000,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// First call - should call LLM and cache result
	provider1, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("First SelectProvider call failed: %v", err)
	}

	if provider1 != "anthropic" {
		t.Errorf("Expected 'anthropic', got '%s'", provider1)
	}

	// Verify LLM was called once
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation, got %d", mockLLM.GetInvocations())
	}

	// Second call immediately - should use cache
	provider2, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("Second SelectProvider call failed: %v", err)
	}

	if provider2 != "anthropic" {
		t.Errorf("Expected cached 'anthropic', got '%s'", provider2)
	}

	// Verify LLM was NOT called again (cache hit)
	if mockLLM.GetInvocations() != 1 {
		t.Errorf("Expected 1 LLM invocation (cache hit), got %d", mockLLM.GetInvocations())
	}

	// Wait for cache to expire
	time.Sleep(shortTTL + 50*time.Millisecond)

	// Third call after TTL - should call LLM again (cache expired)
	provider3, err := strategy.SelectProvider(ctx, reqCtx, providers)
	if err != nil {
		t.Fatalf("Third SelectProvider call failed: %v", err)
	}

	if provider3 != "anthropic" {
		t.Errorf("Expected 'anthropic' after cache expiration, got '%s'", provider3)
	}

	// Verify LLM was called again (cache expired)
	if mockLLM.GetInvocations() != 2 {
		t.Errorf("Expected 2 LLM invocations (cache expired), got %d", mockLLM.GetInvocations())
	}

	t.Logf("✓ Cache expired after %v, LLM called again", shortTTL)
}

// TestLLMRouting_NilLLMModel tests that when no LLM is configured,
// the strategy falls back gracefully.
// Validates: Requirement 11.4
func TestLLMRouting_NilLLMModel(t *testing.T) {
	ctx := context.Background()

	// Create available providers
	providers := map[string]llms.ChatModel{
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
	}

	// Create LLM routing strategy with nil model
	strategy := &LLMRoutingStrategy{
		model:     nil, // No LLM configured
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  1000,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// Call SelectProvider - should fallback without LLM
	provider, err := strategy.SelectProvider(ctx, reqCtx, providers)

	// Error should be returned (indicating LLM not configured)
	if err == nil {
		t.Error("Expected error when LLM is nil, got nil")
	}

	// But a valid fallback provider should still be returned
	if provider == "" {
		t.Fatal("Expected fallback provider, got empty string")
	}

	if _, exists := providers[provider]; !exists {
		t.Errorf("Fallback provider '%s' does not exist in available providers", provider)
	}

	t.Logf("✓ Nil LLM handled gracefully, fell back to provider '%s'", provider)
}

// TestLLMRouting_EmptyProviders tests that when no providers are available,
// an appropriate error is returned.
// Validates: Error handling
func TestLLMRouting_EmptyProviders(t *testing.T) {
	ctx := context.Background()

	// Create mock LLM
	mockLLM := &mockLLMForRouting{
		response:   "openai",
		err:        nil,
		shouldFail: false,
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
		},
		cacheTTL: 5 * time.Minute,
	}

	// Create request context
	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 1,
		TotalTokens:  1000,
		HasToolCalls: false,
		Priority:     "medium",
		Complexity:   "simple",
	}

	// Call SelectProvider with empty providers map
	emptyProviders := map[string]llms.ChatModel{}
	provider, err := strategy.SelectProvider(ctx, reqCtx, emptyProviders)

	// Should return error
	if err == nil {
		t.Error("Expected error when providers map is empty, got nil")
	}

	// Provider should be empty
	if provider != "" {
		t.Errorf("Expected empty provider, got '%s'", provider)
	}

	t.Log("✓ Empty providers map handled correctly with error")
}

// TestLLMRouting_CacheKeyGeneration tests that the cache key generation
// is deterministic and produces different keys for different contexts.
// Validates: Requirement 23.2
func TestLLMRouting_CacheKeyGeneration(t *testing.T) {
	strategy := &LLMRoutingStrategy{
		cacheTTL: 5 * time.Minute,
	}

	t.Run("identical contexts produce identical keys", func(t *testing.T) {
		reqCtx1 := RequestContext{
			MessageCount: 5,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		reqCtx2 := RequestContext{
			MessageCount: 5,
			TotalTokens:  1500,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		key1 := strategy.generateCacheKey(reqCtx1)
		key2 := strategy.generateCacheKey(reqCtx2)

		if key1 != key2 {
			t.Errorf("Expected identical keys for identical contexts, got '%s' vs '%s'", key1, key2)
		}

		if key1 == "" {
			t.Error("Cache key should not be empty")
		}

		t.Logf("✓ Identical contexts produce identical key: %s", key1)
	})

	t.Run("different contexts produce different keys", func(t *testing.T) {
		reqCtx1 := RequestContext{
			MessageCount: 1,
			TotalTokens:  500,
			HasToolCalls: false,
			Priority:     "low",
			Complexity:   "simple",
		}

		reqCtx2 := RequestContext{
			MessageCount: 10,
			TotalTokens:  5000,
			HasToolCalls: true,
			Priority:     "high",
			Complexity:   "complex",
		}

		key1 := strategy.generateCacheKey(reqCtx1)
		key2 := strategy.generateCacheKey(reqCtx2)

		if key1 == key2 {
			t.Errorf("Expected different keys for different contexts, got same key: %s", key1)
		}

		t.Logf("✓ Different contexts produce different keys: %s vs %s", key1, key2)
	})
}

// TestLLMRouting_ProviderNameParsing tests that the parseProviderName function
// correctly extracts provider names from various LLM response formats.
// Validates: Requirement 11.2
func TestLLMRouting_ProviderNameParsing(t *testing.T) {
	strategy := &LLMRoutingStrategy{}

	testCases := []struct {
		name     string
		response string
		expected string
	}{
		{
			name:     "simple provider name",
			response: "openai",
			expected: "openai",
		},
		{
			name:     "provider name with whitespace",
			response: "  anthropic  ",
			expected: "anthropic",
		},
		{
			name:     "provider name with prefix",
			response: "provider: openai",
			expected: "openai",
		},
		{
			name:     "provider name with period",
			response: "anthropic.",
			expected: "anthropic",
		},
		{
			name:     "provider name in quotes",
			response: "\"openai\"",
			expected: "openai",
		},
		{
			name:     "provider name in single quotes",
			response: "'anthropic'",
			expected: "anthropic",
		},
		{
			name:     "provider name in backticks",
			response: "`localmodel`",
			expected: "localmodel",
		},
		{
			name:     "uppercase provider name",
			response: "OPENAI",
			expected: "openai",
		},
		{
			name:     "mixed case provider name",
			response: "OpenAI",
			expected: "openai",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := strategy.parseProviderName(tc.response)

			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}

			t.Logf("✓ Parsed '%s' → '%s'", tc.response, result)
		})
	}
}

// TestLLMRouting_OnSuccessOnError tests that the callback methods
// are no-ops and don't cause errors.
// Validates: Strategy interface implementation
func TestLLMRouting_OnSuccessOnError(t *testing.T) {
	ctx := context.Background()

	strategy := &LLMRoutingStrategy{
		providers: []string{"openai", "anthropic"},
		cacheTTL:  5 * time.Minute,
	}

	// Test OnSuccess - should not panic or error
	strategy.OnSuccess(ctx, "openai", 100*time.Millisecond)
	t.Log("✓ OnSuccess executed without error")

	// Test OnError - should not panic or error
	strategy.OnError(ctx, "anthropic", errors.New("test error"))
	t.Log("✓ OnError executed without error")
}

// TestLLMRouting_PromptBuilding tests that the routing prompt is built correctly
// with request characteristics and provider descriptions.
// Validates: Requirement 11.1
func TestLLMRouting_PromptBuilding(t *testing.T) {
	strategy := &LLMRoutingStrategy{
		providers: []string{"openai", "anthropic", "localmodel"},
		providerDescriptions: map[string]string{
			"openai":     "Fast and cost-effective",
			"anthropic":  "Highly capable for complex tasks",
			"localmodel": "Local model for privacy",
		},
	}

	providers := map[string]llms.ChatModel{
		"openai":     &mockProviderForRouting{name: "openai"},
		"anthropic":  &mockProviderForRouting{name: "anthropic"},
		"localmodel": &mockProviderForRouting{name: "localmodel"},
	}

	reqCtx := RequestContext{
		Messages: []core.Message{
			core.NewHumanMessage("Test message"),
		},
		MessageCount: 5,
		TotalTokens:  2500,
		HasToolCalls: true,
		Priority:     "high",
		Complexity:   "complex",
	}

	prompt := strategy.buildRoutingPrompt(reqCtx, providers)

	// Verify prompt contains request characteristics
	if !strings.Contains(prompt, "Message Count: 5") {
		t.Error("Prompt should contain message count")
	}

	if !strings.Contains(prompt, "Estimated Tokens: 2500") {
		t.Error("Prompt should contain token count")
	}

	if !strings.Contains(prompt, "Has Tool Calls: true") {
		t.Error("Prompt should contain tool calls flag")
	}

	if !strings.Contains(prompt, "Complexity: complex") {
		t.Error("Prompt should contain complexity")
	}

	if !strings.Contains(prompt, "Priority: high") {
		t.Error("Prompt should contain priority")
	}

	// Verify prompt contains provider descriptions
	if !strings.Contains(prompt, "openai: Fast and cost-effective") {
		t.Error("Prompt should contain openai description")
	}

	if !strings.Contains(prompt, "anthropic: Highly capable for complex tasks") {
		t.Error("Prompt should contain anthropic description")
	}

	if !strings.Contains(prompt, "localmodel: Local model for privacy") {
		t.Error("Prompt should contain localmodel description")
	}

	t.Log("✓ Routing prompt built correctly with all required information")
}
