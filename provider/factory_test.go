package provider

import (
	"context"
	"os"
	"testing"
	"testing/quick"

	"github.com/LucaLanziani/langchain-go/llms"
)

// TestProperty1_ProviderCreationConsistency tests that NewProvider returns
// either (model, cleanup, nil) or (nil, nil, error) for all valid provider types.
//
// Property 1: Provider Creation Consistency
// ∀ providerType ∈ ProviderTypes, config ∈ ValidConfigs:
//
//	(model, cleanup, err) := NewProvider(ctx, providerType, config)
//	⟹ (model ≠ nil ∧ cleanup ≠ nil ∧ err = nil) ∨ (model = nil ∧ cleanup = nil ∧ err ≠ nil)
//
// Validates: Requirements 1.1, 1.2, 1.3
func TestProperty1_ProviderCreationConsistency(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name          string
		providerType  ProviderType
		options       []ProviderOption
		shouldSucceed bool
		setupEnv      func()
		cleanupEnv    func()
	}{
		{
			name:         "Anthropic with valid config",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
			shouldSucceed: true,
		},
		{
			name:         "Anthropic with env var",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
			},
			shouldSucceed: true,
			setupEnv: func() {
				os.Setenv("ANTHROPIC_API_KEY", "test-env-key")
			},
			cleanupEnv: func() {
				os.Unsetenv("ANTHROPIC_API_KEY")
			},
		},
		{
			name:         "Anthropic missing MaxTokens",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithAPIKey("test-key"),
			},
			shouldSucceed: false,
		},
		{
			name:         "Anthropic missing auth",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
			},
			shouldSucceed: false,
		},
		{
			name:         "OpenAI with valid config",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
			shouldSucceed: true,
		},
		{
			name:         "OpenAI with env var",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
			},
			shouldSucceed: true,
			setupEnv: func() {
				os.Setenv("OPENAI_API_KEY", "test-env-key")
			},
			cleanupEnv: func() {
				os.Unsetenv("OPENAI_API_KEY")
			},
		},
		{
			name:         "OpenAI missing auth",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
			},
			shouldSucceed: false,
		},
		{
			name:         "Ollama with valid config",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
			},
			shouldSucceed: true,
		},
		{
			name:         "Ollama with custom base URL",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
				WithBaseURL("http://custom:11434"),
			},
			shouldSucceed: true,
		},
		{
			name:         "Ollama with all options",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
				WithTemperature(0.7),
				WithMaxTokens(2000),
				WithTopP(0.9),
				WithStop([]string{"stop1", "stop2"}),
				WithProviderSpecific("keep_alive", "5m"),
				WithProviderSpecific("format", "json"),
			},
			shouldSucceed: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Setup environment if needed
			if tc.setupEnv != nil {
				tc.setupEnv()
			}
			if tc.cleanupEnv != nil {
				defer tc.cleanupEnv()
			}

			// Call NewProvider
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)

			// Verify Property 1: Consistency of return values
			if tc.shouldSucceed {
				// Success case: (model ≠ nil ∧ cleanup ≠ nil ∧ err = nil)
				if model == nil {
					t.Errorf("Expected non-nil model on success, got nil")
				}
				if cleanup == nil {
					t.Errorf("Expected non-nil cleanup on success, got nil")
				}
				if err != nil {
					t.Errorf("Expected nil error on success, got: %v", err)
				}

				// Verify model implements llms.ChatModel interface
				if model != nil {
					var _ llms.ChatModel = model
				}

				// Call cleanup to verify it doesn't panic
				if cleanup != nil {
					cleanupErr := cleanup()
					// Cleanup error is acceptable, but shouldn't panic
					_ = cleanupErr
				}
			} else {
				// Failure case: (model = nil ∧ cleanup = nil ∧ err ≠ nil)
				if model != nil {
					t.Errorf("Expected nil model on failure, got: %v", model)
				}
				if cleanup != nil {
					t.Errorf("Expected nil cleanup on failure, got non-nil function")
				}
				if err == nil {
					t.Errorf("Expected non-nil error on failure, got nil")
				}
			}
		})
	}
}

// TestProperty1_InvalidProviderType tests that invalid provider types
// always return (nil, nil, error)
func TestProperty1_InvalidProviderType(t *testing.T) {
	ctx := context.Background()

	invalidProviderTypes := []ProviderType{
		"invalid",
		"unknown",
		"",
		"aws-bedrock",
		"azure-openai",
	}

	for _, providerType := range invalidProviderTypes {
		t.Run(string(providerType), func(t *testing.T) {
			model, cleanup, err := NewProvider(
				ctx,
				providerType,
				WithModel("test-model"),
			)

			// Must return (nil, nil, error) for invalid provider type
			if model != nil {
				t.Errorf("Expected nil model for invalid provider type, got: %v", model)
			}
			if cleanup != nil {
				t.Errorf("Expected nil cleanup for invalid provider type, got non-nil function")
			}
			if err == nil {
				t.Errorf("Expected non-nil error for invalid provider type, got nil")
			}
		})
	}
}

// TestProperty1_ValidationErrors tests that validation errors
// always return (nil, nil, error)
func TestProperty1_ValidationErrors(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name         string
		providerType ProviderType
		options      []ProviderOption
		description  string
	}{
		{
			name:         "missing model",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithAPIKey("test-key"),
			},
			description: "model name is required",
		},
		{
			name:         "invalid temperature low",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithTemperature(-0.1),
			},
			description: "temperature out of range",
		},
		{
			name:         "invalid temperature high",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithTemperature(2.1),
			},
			description: "temperature out of range",
		},
		{
			name:         "invalid top-p low",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithTopP(-0.1),
			},
			description: "top-p out of range",
		},
		{
			name:         "invalid top-p high",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithTopP(1.1),
			},
			description: "top-p out of range",
		},
		{
			name:         "invalid max tokens",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithMaxTokens(-100),
			},
			description: "max tokens must be positive",
		},
		{
			name:         "invalid max tokens zero",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
				WithMaxTokens(0),
			},
			description: "max tokens must be positive",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)

			// Must return (nil, nil, error) for validation errors
			if model != nil {
				t.Errorf("Expected nil model for validation error (%s), got: %v", tc.description, model)
			}
			if cleanup != nil {
				t.Errorf("Expected nil cleanup for validation error (%s), got non-nil function", tc.description)
			}
			if err == nil {
				t.Errorf("Expected non-nil error for validation error (%s), got nil", tc.description)
			}
		})
	}
}

// TestProperty1_QuickCheck uses property-based testing to verify creation consistency
// with randomly generated valid configurations
func TestProperty1_QuickCheck(t *testing.T) {
	// Skip GitHub Copilot in quick check as it requires actual CLI setup
	providerTypes := []ProviderType{
		ProviderAnthropic,
		ProviderOpenAI,
		ProviderOllama,
	}

	for _, providerType := range providerTypes {
		t.Run(string(providerType), func(t *testing.T) {
			property := func(seed int64) bool {
				ctx := context.Background()

				// Generate valid configuration based on provider type
				options := generateValidConfig(seed, providerType)

				model, cleanup, err := NewProvider(ctx, providerType, options...)

				// Property: Either all success values or all failure values
				successCase := model != nil && cleanup != nil && err == nil
				failureCase := model == nil && cleanup == nil && err != nil

				// Must be exactly one of these cases
				isConsistent := (successCase && !failureCase) || (!successCase && failureCase)

				// Cleanup if successful
				if cleanup != nil {
					_ = cleanup()
				}

				return isConsistent
			}

			config := &quick.Config{MaxCount: 50}
			if err := quick.Check(property, config); err != nil {
				t.Errorf("Property violated for %s: %v", providerType, err)
			}
		})
	}
}

// TestProperty1_CleanupIdempotency tests that cleanup can be called multiple times safely
func TestProperty1_CleanupIdempotency(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name         string
		providerType ProviderType
		options      []ProviderOption
	}{
		{
			name:         "Anthropic",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "OpenAI",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "Ollama",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)
			if err != nil {
				t.Fatalf("Failed to create provider: %v", err)
			}
			if model == nil || cleanup == nil {
				t.Fatal("Expected non-nil model and cleanup")
			}

			// Call cleanup multiple times - should not panic
			for i := 0; i < 3; i++ {
				err := cleanup()
				// Error is acceptable, but shouldn't panic
				_ = err
			}
		})
	}
}

// TestProperty2_CleanupSafety tests that cleanup functions are always safe to call,
// even multiple times, without panicking.
//
// Property 2: Cleanup Safety
// ∀ model, cleanup, err := NewProvider(...):
//
//	err = nil ⟹ cleanup() returns error or nil without panic
//	∧ cleanup(); cleanup() does not panic
//
// Validates: Requirement 5.3
func TestProperty2_CleanupSafety(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name         string
		providerType ProviderType
		options      []ProviderOption
	}{
		{
			name:         "Anthropic cleanup safety",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "OpenAI cleanup safety",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "Ollama cleanup safety",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
		{
			name:         "Ollama with all options cleanup safety",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
				WithBaseURL("http://localhost:11434"),
				WithTemperature(0.7),
				WithMaxTokens(2000),
				WithTopP(0.9),
				WithStop([]string{"stop1", "stop2"}),
				WithProviderSpecific("keep_alive", "5m"),
				WithProviderSpecific("format", "json"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)
			if err != nil {
				t.Fatalf("Failed to create provider: %v", err)
			}
			if model == nil || cleanup == nil {
				t.Fatal("Expected non-nil model and cleanup")
			}

			// Test 1: First cleanup call should not panic
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("First cleanup() panicked: %v", r)
					}
				}()
				err1 := cleanup()
				// Error is acceptable, but shouldn't panic
				_ = err1
			}()

			// Test 2: Second cleanup call should not panic (idempotency)
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("Second cleanup() panicked: %v", r)
					}
				}()
				err2 := cleanup()
				// Error is acceptable, but shouldn't panic
				_ = err2
			}()

			// Test 3: Third cleanup call should not panic (multiple idempotency)
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Errorf("Third cleanup() panicked: %v", r)
					}
				}()
				err3 := cleanup()
				// Error is acceptable, but shouldn't panic
				_ = err3
			}()
		})
	}
}

// TestProperty2_CleanupSafety_ConcurrentCalls tests that cleanup is safe
// when called concurrently from multiple goroutines
func TestProperty2_CleanupSafety_ConcurrentCalls(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name         string
		providerType ProviderType
		options      []ProviderOption
	}{
		{
			name:         "Anthropic concurrent cleanup",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "OpenAI concurrent cleanup",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "Ollama concurrent cleanup",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)
			if err != nil {
				t.Fatalf("Failed to create provider: %v", err)
			}
			if model == nil || cleanup == nil {
				t.Fatal("Expected non-nil model and cleanup")
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
					_ = cleanup()
				}()
			}

			// Wait for all goroutines to complete
			for i := 0; i < numGoroutines; i++ {
				<-done
			}
			close(panicked)

			// Check if any goroutine panicked
			for p := range panicked {
				t.Errorf("Concurrent cleanup() panicked: %v", p)
			}
		})
	}
}

// TestProperty2_CleanupSafety_NilCleanup tests that nil cleanup is never returned on success
func TestProperty2_CleanupSafety_NilCleanup(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name         string
		providerType ProviderType
		options      []ProviderOption
	}{
		{
			name:         "Anthropic returns non-nil cleanup",
			providerType: ProviderAnthropic,
			options: []ProviderOption{
				WithModel("claude-sonnet-4-20250514"),
				WithMaxTokens(4096),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "OpenAI returns non-nil cleanup",
			providerType: ProviderOpenAI,
			options: []ProviderOption{
				WithModel("gpt-4o"),
				WithAPIKey("test-key"),
			},
		},
		{
			name:         "Ollama returns non-nil cleanup",
			providerType: ProviderOllama,
			options: []ProviderOption{
				WithModel("llama3.1"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)
			if err != nil {
				t.Fatalf("Failed to create provider: %v", err)
			}
			if model == nil {
				t.Fatal("Expected non-nil model")
			}

			// Property: cleanup must never be nil on success
			if cleanup == nil {
				t.Error("cleanup function is nil on successful provider creation")
			}
		})
	}
}

// TestProperty2_CleanupSafety_QuickCheck uses property-based testing to verify
// cleanup safety with randomly generated configurations
func TestProperty2_CleanupSafety_QuickCheck(t *testing.T) {
	// Skip GitHub Copilot in quick check as it requires actual CLI setup
	providerTypes := []ProviderType{
		ProviderAnthropic,
		ProviderOpenAI,
		ProviderOllama,
	}

	for _, providerType := range providerTypes {
		t.Run(string(providerType), func(t *testing.T) {
			property := func(seed int64) bool {
				ctx := context.Background()

				// Generate valid configuration
				options := generateValidConfig(seed, providerType)

				model, cleanup, err := NewProvider(ctx, providerType, options...)

				// Only test cleanup safety if provider creation succeeded
				if err != nil || model == nil || cleanup == nil {
					return true // Skip this case
				}

				// Test cleanup safety: should not panic
				panicked := false
				func() {
					defer func() {
						if r := recover(); r != nil {
							panicked = true
						}
					}()
					// Call cleanup multiple times
					_ = cleanup()
					_ = cleanup()
					_ = cleanup()
				}()

				return !panicked
			}

			config := &quick.Config{MaxCount: 50}
			if err := quick.Check(property, config); err != nil {
				t.Errorf("Cleanup safety property violated for %s: %v", providerType, err)
			}
		})
	}
}

// Helper function to generate valid configuration for property-based testing
func generateValidConfig(seed int64, providerType ProviderType) []ProviderOption {
	// Use a simple deterministic approach based on seed
	options := []ProviderOption{
		WithModel("test-model"),
	}

	// Add provider-specific required options
	switch providerType {
	case ProviderAnthropic:
		options = append(options,
			WithMaxTokens(4096),
			WithAPIKey("test-key"),
		)
	case ProviderOpenAI:
		options = append(options,
			WithAPIKey("test-key"),
		)
	case ProviderOllama:
		// Ollama doesn't require auth
	}

	// Add some optional parameters based on seed
	if seed%2 == 0 {
		options = append(options, WithTemperature(0.7))
	}
	if seed%3 == 0 {
		options = append(options, WithTopP(0.9))
	}

	return options
}
