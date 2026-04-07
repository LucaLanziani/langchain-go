package provider

import (
	"context"
	"errors"
	"fmt"
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
}

func (m *mockLLMForRouting) Invoke(ctx context.Context, messages []core.Message, opts ...core.Option) (*core.AIMessage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.invocations++

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
				"openai":    &mockProviderForRouting{name: "openai"},
				"anthropic": &mockProviderForRouting{name: "anthropic"},
				"ollama":    &mockProviderForRouting{name: "ollama"},
			}

			// Create LLM routing strategy
			strategy := &LLMRoutingStrategy{
				model:     mockLLM,
				providers: []string{"openai", "anthropic", "ollama"},
				providerDescriptions: map[string]string{
					"openai":    "Fast and cost-effective",
					"anthropic": "Highly capable",
					"ollama":    "Local model",
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
		"openai":    &mockProviderForRouting{name: "openai"},
		"anthropic": &mockProviderForRouting{name: "anthropic"},
		"ollama":    &mockProviderForRouting{name: "ollama"},
	}

	// Create LLM routing strategy
	strategy := &LLMRoutingStrategy{
		model:     mockLLM,
		providers: []string{"openai", "anthropic", "ollama"},
		providerDescriptions: map[string]string{
			"openai":    "Fast and cost-effective",
			"anthropic": "Highly capable",
			"ollama":    "Local model",
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
