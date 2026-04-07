package provider

import (
	"fmt"
	"os"
)

// defaultConfig returns a ProviderConfig with default values
func defaultConfig() *ProviderConfig {
	return &ProviderConfig{
		ProviderSpecific: make(map[string]any),
	}
}

// Common option functions

// WithModel sets the model name for the provider
func WithModel(model string) ProviderOption {
	return func(c *ProviderConfig) {
		c.Model = model
	}
}

// WithTemperature sets the temperature parameter (0.0 to 2.0)
func WithTemperature(temperature float64) ProviderOption {
	return func(c *ProviderConfig) {
		c.Temperature = &temperature
	}
}

// WithMaxTokens sets the maximum number of tokens to generate
func WithMaxTokens(maxTokens int) ProviderOption {
	return func(c *ProviderConfig) {
		c.MaxTokens = &maxTokens
	}
}

// WithTopP sets the top-p sampling parameter (0.0 to 1.0)
func WithTopP(topP float64) ProviderOption {
	return func(c *ProviderConfig) {
		c.TopP = &topP
	}
}

// WithStop sets the stop sequences
func WithStop(stop []string) ProviderOption {
	return func(c *ProviderConfig) {
		c.Stop = stop
	}
}

// WithAPIKey sets the API key for authentication
func WithAPIKey(apiKey string) ProviderOption {
	return func(c *ProviderConfig) {
		c.APIKey = apiKey
	}
}

// WithBaseURL sets the base URL for the provider API
func WithBaseURL(baseURL string) ProviderOption {
	return func(c *ProviderConfig) {
		c.BaseURL = baseURL
	}
}

// WithProviderSpecific adds a provider-specific configuration option
func WithProviderSpecific(key string, value any) ProviderOption {
	return func(c *ProviderConfig) {
		if c.ProviderSpecific == nil {
			c.ProviderSpecific = make(map[string]any)
		}
		c.ProviderSpecific[key] = value
	}
}

// validateConfig validates the configuration for a specific provider type
func validateConfig(config *ProviderConfig, providerType ProviderType) error {
	// Check common required fields
	if config.Model == "" {
		return fmt.Errorf("model name is required")
	}

	// Validate numeric ranges
	if config.Temperature != nil {
		if *config.Temperature < 0.0 || *config.Temperature > 2.0 {
			return fmt.Errorf("temperature must be between 0.0 and 2.0, got %f", *config.Temperature)
		}
	}

	if config.TopP != nil {
		if *config.TopP < 0.0 || *config.TopP > 1.0 {
			return fmt.Errorf("top-p must be between 0.0 and 1.0, got %f", *config.TopP)
		}
	}

	if config.MaxTokens != nil {
		if *config.MaxTokens <= 0 {
			return fmt.Errorf("max tokens must be positive, got %d", *config.MaxTokens)
		}
	}

	// Check provider-specific requirements
	switch providerType {
	case ProviderAnthropic:
		// Anthropic requires MaxTokens
		if config.MaxTokens == nil || *config.MaxTokens <= 0 {
			return fmt.Errorf("Anthropic provider requires MaxTokens to be set and positive")
		}

		// Check authentication
		if config.APIKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
			return fmt.Errorf("Anthropic provider requires APIKey or ANTHROPIC_API_KEY environment variable")
		}

	case ProviderGitHubCopilot:
		// Check authentication
		if config.APIKey == "" && os.Getenv("GITHUB_TOKEN") == "" {
			// Will try gh CLI as fallback during creation
			// Not an error at validation time
		}

	case ProviderOllama:
		// Ollama doesn't require authentication
		// Set default BaseURL if not provided
		if config.BaseURL == "" {
			config.BaseURL = "http://localhost:11434"
		}

	case ProviderOpenAI:
		// Check authentication
		if config.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
			return fmt.Errorf("OpenAI provider requires APIKey or OPENAI_API_KEY environment variable")
		}

	default:
		return fmt.Errorf("unknown provider type: %s", providerType)
	}

	return nil
}
