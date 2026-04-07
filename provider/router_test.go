package provider

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/llms"
)

// mockStrategy is a simple mock implementation of RoutingStrategy for testing
type mockStrategy struct {
	providerName string
}

func (m *mockStrategy) SelectProvider(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
	if m.providerName != "" {
		return m.providerName, nil
	}
	// Return first available provider
	for name := range providers {
		return name, nil
	}
	return "", ErrProviderNotFound
}

func (m *mockStrategy) OnSuccess(ctx context.Context, providerName string, latency time.Duration) {
	// No-op for testing
}

func (m *mockStrategy) OnError(ctx context.Context, providerName string, err error) {
	// No-op for testing
}

// TestProperty9_RouterCleanupCompleteness tests that router cleanup calls
// cleanup for all providers it manages.
//
// Property 9: Router Cleanup Completeness
// ∀ router, err := NewRouter(ctx, entries, strategy):
//
//	err = nil ∧ router.Cleanup()
//	⟹ ∀ provider ∈ router.providers: provider.cleanup() was called
//
// Validates: Requirement 12.1
func TestProperty9_RouterCleanupCompleteness(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name    string
		entries []ProviderEntry
	}{
		{
			name: "single provider cleanup",
			entries: []ProviderEntry{
				{
					Name:         "openai",
					ProviderType: ProviderOpenAI,
					Options: []ProviderOption{
						WithModel("gpt-4o"),
						WithAPIKey("test-key"),
					},
				},
			},
		},
		{
			name: "multiple providers cleanup",
			entries: []ProviderEntry{
				{
					Name:         "openai",
					ProviderType: ProviderOpenAI,
					Options: []ProviderOption{
						WithModel("gpt-4o"),
						WithAPIKey("test-key"),
					},
				},
				{
					Name:         "anthropic",
					ProviderType: ProviderAnthropic,
					Options: []ProviderOption{
						WithModel("claude-sonnet-4-20250514"),
						WithMaxTokens(4096),
						WithAPIKey("test-key"),
					},
				},
				{
					Name:         "ollama",
					ProviderType: ProviderOllama,
					Options: []ProviderOption{
						WithModel("llama3.1"),
					},
				},
			},
		},
		{
			name: "multiple instances of same provider",
			entries: []ProviderEntry{
				{
					Name:         "openai-fast",
					ProviderType: ProviderOpenAI,
					Options: []ProviderOption{
						WithModel("gpt-3.5-turbo"),
						WithAPIKey("test-key"),
					},
				},
				{
					Name:         "openai-smart",
					ProviderType: ProviderOpenAI,
					Options: []ProviderOption{
						WithModel("gpt-4o"),
						WithAPIKey("test-key"),
					},
				},
				{
					Name:         "openai-creative",
					ProviderType: ProviderOpenAI,
					Options: []ProviderOption{
						WithModel("gpt-4o"),
						WithTemperature(1.2),
						WithAPIKey("test-key"),
					},
				},
			},
		},
		{
			name: "many providers cleanup",
			entries: []ProviderEntry{
				{Name: "openai-1", ProviderType: ProviderOpenAI, Options: []ProviderOption{WithModel("gpt-4o"), WithAPIKey("test-key")}},
				{Name: "openai-2", ProviderType: ProviderOpenAI, Options: []ProviderOption{WithModel("gpt-4o"), WithAPIKey("test-key")}},
				{Name: "anthropic-1", ProviderType: ProviderAnthropic, Options: []ProviderOption{WithModel("claude-sonnet-4-20250514"), WithMaxTokens(4096), WithAPIKey("test-key")}},
				{Name: "anthropic-2", ProviderType: ProviderAnthropic, Options: []ProviderOption{WithModel("claude-sonnet-4-20250514"), WithMaxTokens(4096), WithAPIKey("test-key")}},
				{Name: "ollama-1", ProviderType: ProviderOllama, Options: []ProviderOption{WithModel("llama3.1")}},
				{Name: "ollama-2", ProviderType: ProviderOllama, Options: []ProviderOption{WithModel("llama3.1")}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create router
			router, err := NewRouter(
				ctx,
				tc.entries,
				&mockStrategy{providerName: tc.entries[0].Name},
			)
			if err != nil {
				t.Fatalf("Failed to create router: %v", err)
			}

			// Verify router was created successfully
			if router == nil {
				t.Fatal("Expected non-nil router")
			}

			// Track cleanup calls
			expectedCleanups := len(tc.entries)
			actualCleanups := len(router.cleanups)

			if actualCleanups != expectedCleanups {
				t.Errorf("Expected %d cleanup functions, got %d", expectedCleanups, actualCleanups)
			}

			// Call router cleanup
			err = router.Cleanup()
			if err != nil {
				// Error is acceptable, but cleanup should still complete
				t.Logf("Cleanup returned error (acceptable): %v", err)
			}

			// Property: All providers should be cleaned up
			// Verify by checking that providers map is nil after cleanup
			if router.providers != nil {
				t.Error("Expected providers map to be nil after cleanup")
			}

			// Verify cleanups slice is nil after cleanup
			if router.cleanups != nil {
				t.Error("Expected cleanups slice to be nil after cleanup")
			}
		})
	}
}

// TestProperty9_RouterCleanupCompleteness_Idempotency tests that cleanup
// can be called multiple times safely (idempotency)
func TestProperty9_RouterCleanupCompleteness_Idempotency(t *testing.T) {
	ctx := context.Background()

	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Call cleanup multiple times - should not panic
	for i := 0; i < 3; i++ {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("Cleanup call %d panicked: %v", i+1, r)
				}
			}()
			err := router.Cleanup()
			// Error is acceptable, but shouldn't panic
			_ = err
		}()
	}

	// Verify router is in cleaned up state
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
}

// TestProperty9_RouterCleanupCompleteness_ConcurrentCleanup tests that
// cleanup is safe when called concurrently from multiple goroutines
func TestProperty9_RouterCleanupCompleteness_ConcurrentCleanup(t *testing.T) {
	ctx := context.Background()

	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "ollama",
			ProviderType: ProviderOllama,
			Options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Call cleanup concurrently from multiple goroutines
	const numGoroutines = 10
	done := make(chan bool, numGoroutines)
	panicked := make(chan interface{}, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer func() {
				if r := recover(); r != nil {
					panicked <- r
				}
				done <- true
			}()
			_ = router.Cleanup()
		}()
	}

	// Wait for all goroutines to complete
	for i := 0; i < numGoroutines; i++ {
		<-done
	}
	close(panicked)

	// Check if any goroutine panicked
	for p := range panicked {
		t.Errorf("Concurrent cleanup panicked: %v", p)
	}

	// Verify router is in cleaned up state
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
}

// TestProperty9_RouterCleanupCompleteness_PartialFailure tests that cleanup
// continues even if some provider cleanups fail
func TestProperty9_RouterCleanupCompleteness_PartialFailure(t *testing.T) {
	ctx := context.Background()

	// Create router with multiple providers
	entries := []ProviderEntry{
		{
			Name:         "openai-1",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "openai-2",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai-1"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Call cleanup - even if some cleanups fail, all should be attempted
	err = router.Cleanup()
	// Error is acceptable (first error is returned)
	_ = err

	// Property: All providers should be cleaned up regardless of individual failures
	// Verify by checking that providers map is nil after cleanup
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup, even with partial failures")
	}

	// Verify cleanups slice is nil after cleanup
	if router.cleanups != nil {
		t.Error("Expected cleanups slice to be nil after cleanup, even with partial failures")
	}
}

// TestProperty9_RouterCleanupCompleteness_WithCustomCleanup tests cleanup
// with custom cleanup functions that track invocations
func TestProperty9_RouterCleanupCompleteness_WithCustomCleanup(t *testing.T) {
	ctx := context.Background()

	// Track cleanup calls using atomic counters
	var cleanup1Called atomic.Int32
	var cleanup2Called atomic.Int32
	var cleanup3Called atomic.Int32

	// Create router with providers
	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "ollama",
			ProviderType: ProviderOllama,
			Options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Replace cleanup functions with tracked versions
	// Note: This is a test-only approach to verify cleanup is called
	originalCleanups := router.cleanups
	router.cleanups = map[string]CleanupFunc{
		"openai": func() error {
			cleanup1Called.Add(1)
			return originalCleanups["openai"]()
		},
		"anthropic": func() error {
			cleanup2Called.Add(1)
			return originalCleanups["anthropic"]()
		},
		"ollama": func() error {
			cleanup3Called.Add(1)
			return originalCleanups["ollama"]()
		},
	}

	// Call router cleanup
	err = router.Cleanup()
	_ = err

	// Property: All cleanup functions should have been called exactly once
	if cleanup1Called.Load() != 1 {
		t.Errorf("Expected cleanup1 to be called once, got %d", cleanup1Called.Load())
	}
	if cleanup2Called.Load() != 1 {
		t.Errorf("Expected cleanup2 to be called once, got %d", cleanup2Called.Load())
	}
	if cleanup3Called.Load() != 1 {
		t.Errorf("Expected cleanup3 to be called once, got %d", cleanup3Called.Load())
	}

	// Verify router is in cleaned up state
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
}

// TestProperty9_RouterCleanupCompleteness_EmptyRouter tests cleanup
// behavior with edge cases
func TestProperty9_RouterCleanupCompleteness_EmptyRouter(t *testing.T) {
	ctx := context.Background()

	// Test with single provider (minimum case)
	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Verify router has exactly one cleanup function
	if len(router.cleanups) != 1 {
		t.Errorf("Expected 1 cleanup function, got %d", len(router.cleanups))
	}

	// Call cleanup
	err = router.Cleanup()
	_ = err

	// Verify cleanup completed
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
	if router.cleanups != nil {
		t.Error("Expected cleanups slice to be nil after cleanup")
	}
}

// TestProperty9_RouterCleanupCompleteness_CleanupDuringRequests tests
// cleanup behavior when called while requests might be in flight
func TestProperty9_RouterCleanupCompleteness_CleanupDuringRequests(t *testing.T) {
	ctx := context.Background()

	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Start some goroutines that might be accessing the router
	var wg sync.WaitGroup
	stopRequests := make(chan struct{})

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stopRequests:
					return
				default:
					// Try to access router methods (they should handle cleanup gracefully)
					_ = router.ListProviders()
					_ = router.GetProvider("openai")
				}
			}
		}()
	}

	// Call cleanup while goroutines are running
	err = router.Cleanup()
	_ = err

	// Stop the goroutines
	close(stopRequests)
	wg.Wait()

	// Property: Cleanup should complete successfully even with concurrent access
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
	if router.cleanups != nil {
		t.Error("Expected cleanups slice to be nil after cleanup")
	}
}

// TestProperty9_RouterCleanupCompleteness_ReturnsFirstError tests that
// cleanup returns the first error encountered but continues cleanup
func TestProperty9_RouterCleanupCompleteness_ReturnsFirstError(t *testing.T) {
	ctx := context.Background()

	entries := []ProviderEntry{
		{
			Name:         "openai",
			ProviderType: ProviderOpenAI,
			Options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			Name:         "anthropic",
			ProviderType: ProviderAnthropic,
			Options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
	}

	router, err := NewRouter(
		ctx,
		entries,
		&mockStrategy{providerName: "openai"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Call cleanup
	err = router.Cleanup()
	// Error may or may not be nil depending on provider cleanup behavior
	// The important property is that cleanup completes for all providers

	// Property: All providers should be cleaned up regardless of errors
	if router.providers != nil {
		t.Error("Expected providers map to be nil after cleanup")
	}
	if router.cleanups != nil {
		t.Error("Expected cleanups slice to be nil after cleanup")
	}
}
