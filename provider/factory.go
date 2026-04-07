package provider

import (
	"context"
	"os"
	"os/exec"
	"strings"

	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/providers/anthropic"
	copilot "github.com/LucaLanziani/langchain-go/providers/github-copilot"
	"github.com/LucaLanziani/langchain-go/providers/ollama"
	"github.com/LucaLanziani/langchain-go/providers/openai"
	"github.com/LucaLanziani/langchain-go/tools"
)

// NewProvider creates a ChatModel for the specified provider type with unified configuration.
// Returns the created model, a cleanup function, and an error if creation fails.
//
// The cleanup function must be called when the model is no longer needed to release resources.
// For most providers (Anthropic, OpenAI, Ollama), cleanup is a no-op. For GitHub Copilot,
// cleanup stops the CLI server process.
//
// Example:
//
//	model, cleanup, err := provider.NewProvider(ctx, provider.ProviderOpenAI,
//	    provider.WithModel("gpt-4o"),
//	    provider.WithTemperature(0.7),
//	    provider.WithAPIKey("sk-..."),
//	)
//	if err != nil {
//	    return err
//	}
//	defer cleanup()
func NewProvider(ctx context.Context, providerType ProviderType, opts ...ProviderOption) (llms.ChatModel, CleanupFunc, error) {
	// Step 1: Build unified configuration
	config := defaultConfig()
	for _, opt := range opts {
		opt(config)
	}

	// Step 2: Validate configuration
	if err := validateConfig(config, providerType); err != nil {
		return nil, nil, NewProviderError(providerType, "", "validation", err)
	}

	// Step 3: Create provider-specific instance
	switch providerType {
	case ProviderAnthropic:
		return createAnthropic(config)
	case ProviderGitHubCopilot:
		return createCopilot(ctx, config)
	case ProviderOllama:
		return createOllama(config)
	case ProviderOpenAI:
		return createOpenAI(config)
	default:
		return nil, nil, NewProviderError(providerType, "", "creation", ErrUnknownProvider)
	}
}

// createAnthropic creates an Anthropic ChatModel from unified configuration
func createAnthropic(config *ProviderConfig) (llms.ChatModel, CleanupFunc, error) {
	opts := []anthropic.OptionFunc{}

	// Authentication
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, anthropic.WithAPIKey(apiKey))
	}

	// Model
	if config.Model != "" {
		opts = append(opts, anthropic.WithModelName(config.Model))
	}

	// Base URL
	if config.BaseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(config.BaseURL))
	}

	// MaxTokens (required by Anthropic)
	if config.MaxTokens != nil {
		opts = append(opts, anthropic.WithMaxTokens(*config.MaxTokens))
	}

	model := anthropic.New(opts...)

	// Anthropic doesn't need cleanup
	cleanup := func() error { return nil }

	return model, cleanup, nil
}

// createCopilot creates a GitHub Copilot ChatModel from unified configuration
func createCopilot(ctx context.Context, config *ProviderConfig) (llms.ChatModel, CleanupFunc, error) {
	opts := []copilot.OptionFunc{}

	// Authentication
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GITHUB_TOKEN")
	}
	if apiKey == "" {
		// Try gh CLI as fallback
		if out, err := exec.Command("gh", "auth", "token").Output(); err == nil {
			apiKey = strings.TrimSpace(string(out))
		}
	}
	if apiKey != "" {
		opts = append(opts, copilot.WithGithubToken(apiKey))
	}

	// Model
	if config.Model != "" {
		opts = append(opts, copilot.WithModelName(config.Model))
	}

	// Provider-specific options
	if cliPath, ok := config.ProviderSpecific["cli_path"].(string); ok {
		opts = append(opts, copilot.WithCLIPath(cliPath))
	}

	if logLevel, ok := config.ProviderSpecific["log_level"].(string); ok {
		opts = append(opts, copilot.WithLogLevel(logLevel))
	}

	if maxConc, ok := config.ProviderSpecific["max_concurrency"].(int); ok {
		opts = append(opts, copilot.WithMaxConcurrency(maxConc))
	}

	// Tools
	if toolsVal, ok := config.ProviderSpecific["tools"]; ok {
		if toolsSlice, ok := toolsVal.([]tools.Tool); ok {
			opts = append(opts, copilot.WithTools(toolsSlice...))
		}
	}

	model, err := copilot.New(ctx, opts...)
	if err != nil {
		return nil, nil, NewProviderError(ProviderGitHubCopilot, "", "creation", err)
	}

	// Copilot needs explicit cleanup to stop CLI server
	cleanup := func() error {
		return model.Close()
	}

	return model, cleanup, nil
}

// createOllama creates an Ollama ChatModel from unified configuration
func createOllama(config *ProviderConfig) (llms.ChatModel, CleanupFunc, error) {
	opts := []ollama.OptionFunc{}

	// Model
	if config.Model != "" {
		opts = append(opts, ollama.WithModel(config.Model))
	}

	// Base URL
	if config.BaseURL != "" {
		opts = append(opts, ollama.WithBaseURL(config.BaseURL))
	}

	// Temperature
	if config.Temperature != nil {
		opts = append(opts, ollama.WithTemperature(*config.Temperature))
	}

	// TopP
	if config.TopP != nil {
		opts = append(opts, ollama.WithTopP(*config.TopP))
	}

	// MaxTokens (maps to NumPredict in Ollama)
	if config.MaxTokens != nil {
		opts = append(opts, ollama.WithNumPredict(*config.MaxTokens))
	}

	// Stop sequences
	if len(config.Stop) > 0 {
		opts = append(opts, ollama.WithStop(config.Stop))
	}

	// Provider-specific options
	if keepAlive, ok := config.ProviderSpecific["keep_alive"].(string); ok {
		opts = append(opts, ollama.WithKeepAlive(keepAlive))
	}

	if format, ok := config.ProviderSpecific["format"].(string); ok {
		opts = append(opts, ollama.WithFormat(format))
	}

	if numCtx, ok := config.ProviderSpecific["num_ctx"].(int); ok {
		opts = append(opts, ollama.WithNumCtx(numCtx))
	}

	if topK, ok := config.ProviderSpecific["top_k"].(int); ok {
		opts = append(opts, ollama.WithTopK(topK))
	}

	model := ollama.New(opts...)

	// Ollama doesn't need cleanup
	cleanup := func() error { return nil }

	return model, cleanup, nil
}

// createOpenAI creates an OpenAI ChatModel from unified configuration
func createOpenAI(config *ProviderConfig) (llms.ChatModel, CleanupFunc, error) {
	opts := []openai.OptionFunc{}

	// Authentication
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey != "" {
		opts = append(opts, openai.WithAPIKey(apiKey))
	}

	// Model
	if config.Model != "" {
		opts = append(opts, openai.WithModelName(config.Model))
	}

	// Base URL
	if config.BaseURL != "" {
		opts = append(opts, openai.WithBaseURL(config.BaseURL))
	}

	// Provider-specific options
	if org, ok := config.ProviderSpecific["organization"].(string); ok {
		opts = append(opts, openai.WithOrganization(org))
	}

	model := openai.New(opts...)

	// OpenAI doesn't need cleanup
	cleanup := func() error { return nil }

	return model, cleanup, nil
}

// noOpCleanup is a cleanup function that does nothing
func noOpCleanup() error {
	return nil
}
