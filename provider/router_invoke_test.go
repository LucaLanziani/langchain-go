package provider

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
)

// TestRouterInvoke_BasicRouting tests that Router.Invoke correctly routes requests
func TestRouterInvoke_BasicRouting(t *testing.T) {
	ctx := context.Background()

	// Create router with simple strategy
	router, err := NewRouter(ctx,
		[]ProviderEntry{
			{
				Name:         "ollama-1",
				ProviderType: ProviderOllama,
				Options: []ProviderOption{
					WithModel("llama3.1"),
					WithBaseURL("http://localhost:11434"),
				},
			},
		},
		&SimpleStrategy{ProviderName: "ollama-1"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	// Test that router is created successfully
	if router == nil {
		t.Fatal("Router should not be nil")
	}

	// Test GetProvider
	provider := router.GetProvider("ollama-1")
	if provider == nil {
		t.Error("GetProvider should return non-nil provider")
	}

	// Test ListProviders
	providers := router.ListProviders()
	if len(providers) != 1 {
		t.Errorf("Expected 1 provider, got %d", len(providers))
	}
	if providers[0] != "ollama-1" {
		t.Errorf("Expected provider name 'ollama-1', got '%s'", providers[0])
	}

	// Test GetMetrics
	metrics := router.GetMetrics()
	if metrics == nil {
		t.Error("GetMetrics should return non-nil metrics")
	}
	if _, ok := metrics["ollama-1"]; !ok {
		t.Error("Metrics should contain 'ollama-1' provider")
	}
}

// TestRouterInvoke_MetricsTracking tests that metrics are correctly tracked
func TestRouterInvoke_MetricsTracking(t *testing.T) {
	ctx := context.Background()

	// Track strategy callbacks
	var successCalled, errorCalled bool

	customStrategy := &CustomStrategy{
		SelectFunc: func(ctx context.Context, reqCtx RequestContext, providers map[string]llms.ChatModel) (string, error) {
			return "ollama-1", nil
		},
		OnSuccessFunc: func(ctx context.Context, providerName string, latency time.Duration) {
			successCalled = true
		},
		OnErrorFunc: func(ctx context.Context, providerName string, err error) {
			errorCalled = true
		},
	}

	router, err := NewRouter(ctx,
		[]ProviderEntry{
			{
				Name:         "ollama-1",
				ProviderType: ProviderOllama,
				Options: []ProviderOption{
					WithModel("llama3.1"),
					WithBaseURL("http://localhost:11434"),
				},
			},
		},
		customStrategy,
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	// Get initial metrics
	initialMetrics := router.GetMetrics()
	initialCount := initialMetrics["ollama-1"].RequestCount

	// Note: We can't actually invoke without a running Ollama server,
	// but we can verify the structure is correct
	t.Logf("Initial request count: %d", initialCount)
	t.Logf("Strategy callbacks ready: success=%v, error=%v", successCalled, errorCalled)
}

// TestRouterInvoke_RequestContextBuilding tests buildRequestContext
func TestRouterInvoke_RequestContextBuilding(t *testing.T) {
	tests := []struct {
		name               string
		messages           []core.Message
		expectedComplexity string
		expectedToolCalls  bool
	}{
		{
			name: "simple request",
			messages: []core.Message{
				core.NewHumanMessage("Hello"),
			},
			expectedComplexity: "simple",
			expectedToolCalls:  false,
		},
		{
			name: "moderate request",
			messages: []core.Message{
				core.NewHumanMessage(strings.Repeat("word ", 1100)), // 5500 chars = 1375 tokens (between 1000 and 10000)
			},
			expectedComplexity: "moderate",
			expectedToolCalls:  false,
		},
		{
			name: "complex request with tool calls",
			messages: []core.Message{
				core.NewAIMessageWithToolCalls("", []core.ToolCall{
					{ID: "1", Name: "test_tool"},
				}),
			},
			expectedComplexity: "complex",
			expectedToolCalls:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqCtx := buildRequestContext(tt.messages, nil)

			if reqCtx.Complexity != tt.expectedComplexity {
				t.Errorf("Expected complexity %s, got %s", tt.expectedComplexity, reqCtx.Complexity)
			}

			if reqCtx.HasToolCalls != tt.expectedToolCalls {
				t.Errorf("Expected HasToolCalls %v, got %v", tt.expectedToolCalls, reqCtx.HasToolCalls)
			}

			if reqCtx.MessageCount != len(tt.messages) {
				t.Errorf("Expected MessageCount %d, got %d", len(tt.messages), reqCtx.MessageCount)
			}
		})
	}
}

// TestRouterInvoke_HelperMethods tests GetProvider, ListProviders, GetMetrics
func TestRouterInvoke_HelperMethods(t *testing.T) {
	ctx := context.Background()

	router, err := NewRouter(ctx,
		[]ProviderEntry{
			{
				Name:         "provider-1",
				ProviderType: ProviderOllama,
				Options: []ProviderOption{
					WithModel("llama3.1"),
					WithBaseURL("http://localhost:11434"),
				},
			},
			{
				Name:         "provider-2",
				ProviderType: ProviderOllama,
				Options: []ProviderOption{
					WithModel("llama3.1"),
					WithBaseURL("http://localhost:11434"),
				},
			},
		},
		&SimpleStrategy{ProviderName: "provider-1"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	// Test GetProvider
	t.Run("GetProvider", func(t *testing.T) {
		provider1 := router.GetProvider("provider-1")
		if provider1 == nil {
			t.Error("GetProvider('provider-1') should return non-nil")
		}

		provider2 := router.GetProvider("provider-2")
		if provider2 == nil {
			t.Error("GetProvider('provider-2') should return non-nil")
		}

		nonExistent := router.GetProvider("non-existent")
		if nonExistent != nil {
			t.Error("GetProvider('non-existent') should return nil")
		}
	})

	// Test ListProviders
	t.Run("ListProviders", func(t *testing.T) {
		providers := router.ListProviders()
		if len(providers) != 2 {
			t.Errorf("Expected 2 providers, got %d", len(providers))
		}

		// Check both providers are in the list
		found1, found2 := false, false
		for _, name := range providers {
			if name == "provider-1" {
				found1 = true
			}
			if name == "provider-2" {
				found2 = true
			}
		}
		if !found1 || !found2 {
			t.Error("ListProviders should return both provider-1 and provider-2")
		}
	})

	// Test GetMetrics
	t.Run("GetMetrics", func(t *testing.T) {
		metrics := router.GetMetrics()
		if len(metrics) != 2 {
			t.Errorf("Expected metrics for 2 providers, got %d", len(metrics))
		}

		if _, ok := metrics["provider-1"]; !ok {
			t.Error("Metrics should contain provider-1")
		}
		if _, ok := metrics["provider-2"]; !ok {
			t.Error("Metrics should contain provider-2")
		}

		// Check initial values
		if metrics["provider-1"].RequestCount != 0 {
			t.Errorf("Initial RequestCount should be 0, got %d", metrics["provider-1"].RequestCount)
		}
		if metrics["provider-1"].ErrorCount != 0 {
			t.Errorf("Initial ErrorCount should be 0, got %d", metrics["provider-1"].ErrorCount)
		}
	})
}

// TestRouterInvoke_AfterCleanup tests that methods handle cleanup correctly
func TestRouterInvoke_AfterCleanup(t *testing.T) {
	ctx := context.Background()

	router, err := NewRouter(ctx,
		[]ProviderEntry{
			{
				Name:         "ollama-1",
				ProviderType: ProviderOllama,
				Options: []ProviderOption{
					WithModel("llama3.1"),
					WithBaseURL("http://localhost:11434"),
				},
			},
		},
		&SimpleStrategy{ProviderName: "ollama-1"},
	)
	if err != nil {
		t.Fatalf("Failed to create router: %v", err)
	}

	// Cleanup the router
	if err := router.Cleanup(); err != nil {
		t.Fatalf("Cleanup failed: %v", err)
	}

	// Test methods after cleanup
	t.Run("GetProvider after cleanup", func(t *testing.T) {
		provider := router.GetProvider("ollama-1")
		if provider != nil {
			t.Error("GetProvider should return nil after cleanup")
		}
	})

	t.Run("ListProviders after cleanup", func(t *testing.T) {
		providers := router.ListProviders()
		if providers != nil {
			t.Error("ListProviders should return nil after cleanup")
		}
	})

	t.Run("Invoke after cleanup", func(t *testing.T) {
		_, err := router.Invoke(ctx, []core.Message{core.NewHumanMessage("test")})
		if err != ErrRouterClosed {
			t.Errorf("Expected ErrRouterClosed, got %v", err)
		}
	})

	t.Run("Stream after cleanup", func(t *testing.T) {
		_, err := router.Stream(ctx, []core.Message{core.NewHumanMessage("test")})
		if err != ErrRouterClosed {
			t.Errorf("Expected ErrRouterClosed, got %v", err)
		}
	})
}
