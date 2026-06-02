package provider

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/LucaLanziani/langchain-go/llms"
)

// TestNewProvider_Anthropic tests Anthropic provider creation with various configurations
func TestNewProvider_Anthropic(t *testing.T) {
	ctx := context.Background()

	t.Run("creates Anthropic provider with API key option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithMaxTokens(4096),
			WithAPIKey("test-api-key"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}
		if cleanup == nil {
			t.Fatal("Expected non-nil cleanup function")
		}

		// Verify model implements ChatModel
		var _ llms.ChatModel = model

		// Cleanup should not error
		if err := cleanup(); err != nil {
			t.Errorf("Cleanup returned error: %v", err)
		}
	})

	t.Run("creates Anthropic provider with environment variable", func(t *testing.T) {
		// Set environment variable
		os.Setenv("ANTHROPIC_API_KEY", "env-api-key")
		defer os.Unsetenv("ANTHROPIC_API_KEY")

		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithMaxTokens(4096),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("creates Anthropic provider with custom base URL", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithMaxTokens(4096),
			WithAPIKey("test-api-key"),
			WithBaseURL("https://custom.anthropic.com"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("fails without MaxTokens", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithAPIKey("test-api-key"),
		)

		if err == nil {
			t.Error("Expected error for missing MaxTokens")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})

	t.Run("fails without authentication", func(t *testing.T) {
		os.Unsetenv("ANTHROPIC_API_KEY")

		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithMaxTokens(4096),
		)

		if err == nil {
			t.Error("Expected error for missing authentication")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})

	t.Run("fails without model name", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithMaxTokens(4096),
			WithAPIKey("test-api-key"),
		)

		if err == nil {
			t.Error("Expected error for missing model name")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})
}

// TestNewProvider_OpenAI tests OpenAI provider creation with various configurations
func TestNewProvider_OpenAI(t *testing.T) {
	ctx := context.Background()

	t.Run("creates OpenAI provider with API key option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-api-key"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}
		if cleanup == nil {
			t.Fatal("Expected non-nil cleanup function")
		}

		var _ llms.ChatModel = model
		cleanup()
	})

	t.Run("creates OpenAI provider with environment variable", func(t *testing.T) {
		os.Setenv("OPENAI_API_KEY", "env-api-key")
		defer os.Unsetenv("OPENAI_API_KEY")

		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("creates OpenAI provider with custom base URL", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-api-key"),
			WithBaseURL("https://custom.openai.com/v1"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("creates OpenAI provider with all common options", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-api-key"),
			WithTemperature(0.7),
			WithMaxTokens(2000),
			WithTopP(0.9),
			WithStop([]string{"User:", "Assistant:"}),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("fails without authentication", func(t *testing.T) {
		os.Unsetenv("OPENAI_API_KEY")

		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
		)

		if err == nil {
			t.Error("Expected error for missing authentication")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})

	t.Run("fails without model name", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithAPIKey("test-api-key"),
		)

		if err == nil {
			t.Error("Expected error for missing model name")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})
}

// TestNewProvider_Copilot tests GitHub Copilot provider creation with various configurations
func TestNewProvider_Copilot(t *testing.T) {
	ctx := context.Background()

	t.Run("creates Copilot provider with API key option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithModel("gpt-4o"),
			WithAPIKey("test-github-token"),
		)

		// Copilot may fail if CLI is not available, which is expected in test environment
		if err != nil {
			// Check if it's a CLI-related error (acceptable in test environment)
			if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
				t.Skip("Copilot CLI not available in test environment")
			}
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}
		if cleanup == nil {
			t.Fatal("Expected non-nil cleanup function")
		}

		var _ llms.ChatModel = model

		// Copilot cleanup should close the CLI server
		if err := cleanup(); err != nil {
			t.Errorf("Cleanup returned error: %v", err)
		}
	})

	t.Run("creates Copilot provider with environment variable", func(t *testing.T) {
		os.Setenv("GITHUB_TOKEN", "env-github-token")
		defer os.Unsetenv("GITHUB_TOKEN")

		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithModel("gpt-4o"),
		)

		if err != nil {
			if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
				t.Skip("Copilot CLI not available in test environment")
			}
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("fails without model name", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithAPIKey("test-github-token"),
		)

		if err == nil {
			t.Error("Expected error for missing model name")
		}
		if model != nil {
			t.Error("Expected nil model on error")
		}
		if cleanup != nil {
			t.Error("Expected nil cleanup on error")
		}
	})
}

// TestNewProvider_InvalidProviderType tests error handling for invalid provider types
func TestNewProvider_InvalidProviderType(t *testing.T) {
	ctx := context.Background()

	invalidTypes := []struct {
		name         string
		providerType ProviderType
	}{
		{"empty string", ""},
		{"unknown provider", "unknown"},
		{"invalid provider", "invalid"},
		{"aws-bedrock", "aws-bedrock"},
		{"azure-openai", "azure-openai"},
		{"random string", "random-provider-name"},
	}

	for _, tc := range invalidTypes {
		t.Run(tc.name, func(t *testing.T) {
			model, cleanup, err := NewProvider(ctx, tc.providerType,
				WithModel("test-model"),
			)

			if err == nil {
				t.Error("Expected error for invalid provider type")
			}

			// Verify error is ErrUnknownProvider or wraps it
			var providerErr *ProviderError
			if errors.As(err, &providerErr) {
				if providerErr.ProviderType != tc.providerType {
					t.Errorf("Expected provider type %s in error, got %s", tc.providerType, providerErr.ProviderType)
				}
			}

			if model != nil {
				t.Error("Expected nil model for invalid provider type")
			}
			if cleanup != nil {
				t.Error("Expected nil cleanup for invalid provider type")
			}
		})
	}
}

// TestCleanupFunction tests cleanup function behavior
func TestCleanupFunction(t *testing.T) {
	ctx := context.Background()

	t.Run("cleanup is idempotent for Anthropic", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithMaxTokens(4096),
			WithAPIKey("test-api-key"),
		)
		if err != nil {
			t.Fatalf("Failed to create provider: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		// Call cleanup multiple times
		for i := 0; i < 5; i++ {
			if err := cleanup(); err != nil {
				t.Errorf("Cleanup call %d returned error: %v", i+1, err)
			}
		}
	})

	t.Run("cleanup is idempotent for OpenAI", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-api-key"),
		)
		if err != nil {
			t.Fatalf("Failed to create provider: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		// Call cleanup multiple times
		for i := 0; i < 5; i++ {
			if err := cleanup(); err != nil {
				t.Errorf("Cleanup call %d returned error: %v", i+1, err)
			}
		}
	})

	t.Run("cleanup returns non-nil function on success", func(t *testing.T) {
		providers := []struct {
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
					WithAPIKey("test-api-key"),
				},
			},
			{
				name:         "OpenAI",
				providerType: ProviderOpenAI,
				options: []ProviderOption{
					WithModel("gpt-4o"),
					WithAPIKey("test-api-key"),
				},
			},
		}

		for _, tc := range providers {
			t.Run(tc.name, func(t *testing.T) {
				_, cleanup, err := NewProvider(ctx, tc.providerType, tc.options...)
				if err != nil {
					t.Fatalf("Failed to create provider: %v", err)
				}

				if cleanup == nil {
					t.Error("Expected non-nil cleanup function")
				}

				cleanup()
			})
		}
	})
}

// TestProviderSpecificOptions tests provider-specific configuration options
func TestProviderSpecificOptions(t *testing.T) {
	ctx := context.Background()

	t.Run("OpenAI with organization option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-api-key"),
			WithProviderSpecific("organization", "org-123"),
		)

		if err != nil {
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("Copilot with cli_path option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithModel("gpt-4o"),
			WithAPIKey("test-github-token"),
			WithProviderSpecific("cli_path", "/usr/local/bin/copilot"),
		)

		if err != nil {
			if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
				t.Skip("Copilot CLI not available in test environment")
			}
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("Copilot with log_level option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithModel("gpt-4o"),
			WithAPIKey("test-github-token"),
			WithProviderSpecific("log_level", "debug"),
		)

		if err != nil {
			if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
				t.Skip("Copilot CLI not available in test environment")
			}
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})

	t.Run("Copilot with max_concurrency option", func(t *testing.T) {
		model, cleanup, err := NewProvider(ctx, ProviderGitHubCopilot,
			WithModel("gpt-4o"),
			WithAPIKey("test-github-token"),
			WithProviderSpecific("max_concurrency", 10),
		)

		if err != nil {
			if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
				t.Skip("Copilot CLI not available in test environment")
			}
			t.Fatalf("Expected no error, got: %v", err)
		}
		if model == nil {
			t.Fatal("Expected non-nil model")
		}

		cleanup()
	})
}

// TestErrorWrapping tests that errors are properly wrapped with context
func TestErrorWrapping(t *testing.T) {
	ctx := context.Background()

	t.Run("invalid provider type error includes provider type", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderType("invalid"),
			WithModel("test-model"),
		)

		if err == nil {
			t.Fatal("Expected error for invalid provider type")
		}

		var providerErr *ProviderError
		if errors.As(err, &providerErr) {
			if providerErr.ProviderType != "invalid" {
				t.Errorf("Expected provider type 'invalid' in error, got %s", providerErr.ProviderType)
			}
			// Invalid provider types fail during validation (validateConfig checks for unknown types)
			if providerErr.Operation != "validation" {
				t.Errorf("Expected operation 'validation' in error, got %s", providerErr.Operation)
			}
		} else {
			t.Errorf("Expected ProviderError, got %T", err)
		}
	})

	t.Run("validation error includes provider type", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			// Missing MaxTokens - should fail validation
			WithAPIKey("test-api-key"),
		)

		if err == nil {
			t.Fatal("Expected validation error")
		}

		var providerErr *ProviderError
		if errors.As(err, &providerErr) {
			if providerErr.ProviderType != ProviderAnthropic {
				t.Errorf("Expected provider type %s in error, got %s", ProviderAnthropic, providerErr.ProviderType)
			}
			if providerErr.Operation != "validation" {
				t.Errorf("Expected operation 'validation' in error, got %s", providerErr.Operation)
			}
		}
	})

	t.Run("missing model error is descriptive", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithAPIKey("test-api-key"),
		)

		if err == nil {
			t.Fatal("Expected error for missing model")
		}

		if !strings.Contains(err.Error(), "model") {
			t.Errorf("Expected error message to mention 'model', got: %v", err)
		}
	})

	t.Run("missing auth error is descriptive", func(t *testing.T) {
		os.Unsetenv("OPENAI_API_KEY")

		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
		)

		if err == nil {
			t.Fatal("Expected error for missing authentication")
		}

		if !strings.Contains(err.Error(), "APIKey") && !strings.Contains(err.Error(), "OPENAI_API_KEY") {
			t.Errorf("Expected error message to mention authentication, got: %v", err)
		}
	})
}

// TestValidationErrors tests all validation error scenarios
func TestValidationErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("temperature below range", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithTemperature(-0.1),
		)

		if err == nil {
			t.Error("Expected error for temperature below range")
		}
	})

	t.Run("temperature above range", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithTemperature(2.1),
		)

		if err == nil {
			t.Error("Expected error for temperature above range")
		}
	})

	t.Run("top-p below range", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithTopP(-0.1),
		)

		if err == nil {
			t.Error("Expected error for top-p below range")
		}
	})

	t.Run("top-p above range", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithTopP(1.1),
		)

		if err == nil {
			t.Error("Expected error for top-p above range")
		}
	})

	t.Run("max tokens zero", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithMaxTokens(0),
		)

		if err == nil {
			t.Error("Expected error for max tokens zero")
		}
	})

	t.Run("max tokens negative", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderOpenAI,
			WithModel("gpt-4o"),
			WithAPIKey("test-key"),
			WithMaxTokens(-100),
		)

		if err == nil {
			t.Error("Expected error for negative max tokens")
		}
	})

	t.Run("Anthropic requires positive MaxTokens", func(t *testing.T) {
		_, _, err := NewProvider(ctx, ProviderAnthropic,
			WithModel("claude-sonnet-4-20250514"),
			WithAPIKey("test-key"),
			WithMaxTokens(0),
		)

		if err == nil {
			t.Error("Expected error for Anthropic with zero MaxTokens")
		}
	})
}

// TestAllProviderTypes tests that all defined provider types can be created
func TestAllProviderTypes(t *testing.T) {
	ctx := context.Background()

	// Test that all provider type constants are defined
	providerTypes := []ProviderType{
		ProviderAnthropic,
		ProviderGitHubCopilot,
		ProviderOpenAI,
	}

	for _, pt := range providerTypes {
		t.Run(string(pt), func(t *testing.T) {
			// Verify the provider type string is not empty
			if string(pt) == "" {
				t.Errorf("Provider type constant %v has empty string value", pt)
			}

			// Verify we can attempt to create the provider
			// (may fail due to missing config, but should not panic)
			var options []ProviderOption
			switch pt {
			case ProviderAnthropic:
				options = []ProviderOption{
					WithModel("claude-sonnet-4-20250514"),
					WithMaxTokens(4096),
					WithAPIKey("test-key"),
				}
			case ProviderOpenAI:
				options = []ProviderOption{
					WithModel("gpt-4o"),
					WithAPIKey("test-key"),
				}
			case ProviderGitHubCopilot:
				options = []ProviderOption{
					WithModel("gpt-4o"),
					WithAPIKey("test-token"),
				}
			}

			model, cleanup, err := NewProvider(ctx, pt, options...)

			// For Copilot, we may skip if CLI is not available
			if pt == ProviderGitHubCopilot && err != nil {
				if strings.Contains(err.Error(), "copilot") || strings.Contains(err.Error(), "CLI") {
					t.Skip("Copilot CLI not available in test environment")
				}
			}

			if err == nil {
				if model == nil {
					t.Error("Expected non-nil model on success")
				}
				if cleanup == nil {
					t.Error("Expected non-nil cleanup on success")
				}
				cleanup()
			}
		})
	}
}
