// Package copilot provides a GitHub Copilot chat model implementation using the Copilot SDK.
package copilot

import (
	copilot "github.com/github/copilot-sdk/go"

	"github.com/LucaLanziani/langchain-go/tools"
)

// Options holds configuration for the GitHub Copilot chat model.
type Options struct {
	// GithubToken is the GitHub token for authentication. Falls back to GITHUB_TOKEN env var.
	GithubToken string

	// Model is the model ID (e.g., "gpt-5-mini", "gpt-5", "claude-sonnet-4.5").
	Model string

	// CLIPath is the path to the Copilot CLI executable. Defaults to "copilot".
	CLIPath string

	// LogLevel for the CLI server (e.g., "error", "info", "debug").
	LogLevel string

	// MaxConcurrency controls the maximum number of parallel sessions in Batch.
	// Defaults to 5.
	MaxConcurrency int

	// Temperature controls randomness (0.0 to 2.0).
	Temperature *float64

	// MaxTokens limits the response length.
	MaxTokens *int

	// TopP controls nucleus sampling.
	TopP *float64

	// Stop sequences.
	Stop []string

	// Tools are langchain Tool implementations that get bridged to SDK tool handlers.
	// When set, the SDK manages the full tool-calling loop internally.
	Tools []tools.Tool

	// ReasoningEffort controls the effort the model spends on reasoning.
	// Valid values: "low", "medium", "high", "xhigh".
	// Only applies to models that support reasoning (e.g. gpt-5-mini, gpt-5).
	// Empty string (default) disables the field and uses the model default.
	ReasoningEffort string

	// OnPermissionRequest is a handler for permission requests from the server.
	// If nil, all permission requests are denied by default.
	// Provide a handler to approve operations (file writes, shell commands, URL fetches, etc.).
	OnPermissionRequest copilot.PermissionHandlerFunc
}

// DefaultOptions returns sensible defaults for the GitHub Copilot provider.
func DefaultOptions() *Options {
	return &Options{
		Model:          "gpt-5-mini",
		LogLevel:       "error",
		MaxConcurrency: 5,
	}
}

// OptionFunc configures Copilot-specific options.
type OptionFunc func(*Options)

// WithGithubToken sets the GitHub token for authentication.
func WithGithubToken(token string) OptionFunc {
	return func(o *Options) { o.GithubToken = token }
}

// WithModelName sets the model name.
func WithModelName(model string) OptionFunc {
	return func(o *Options) { o.Model = model }
}

// WithCLIPath sets the path to the Copilot CLI executable.
func WithCLIPath(path string) OptionFunc {
	return func(o *Options) { o.CLIPath = path }
}

// WithLogLevel sets the log level for the CLI server.
func WithLogLevel(level string) OptionFunc {
	return func(o *Options) { o.LogLevel = level }
}

// WithMaxConcurrency sets the maximum number of parallel sessions in Batch.
func WithMaxConcurrency(n int) OptionFunc {
	return func(o *Options) { o.MaxConcurrency = n }
}

// WithTools sets langchain Tool implementations that get bridged to SDK tool handlers.
// The SDK manages the full tool-calling loop internally, so Invoke returns the
// final response after all tool calls are resolved.
func WithTools(t ...tools.Tool) OptionFunc {
	return func(o *Options) { o.Tools = append(o.Tools, t...) }
}

// WithReasoningEffort sets the reasoning effort level for models that support it.
// Valid values: "low", "medium", "high", "xhigh".
// When set, the model will emit reasoning_delta events alongside message_delta events
// during streaming, and ThinkingFunc in Jarvis Config will be called with each token.
func WithReasoningEffort(effort string) OptionFunc {
	return func(o *Options) { o.ReasoningEffort = effort }
}

// WithPermissionHandler sets a custom permission handler for approving operations.
// If not set, all permission requests are denied by default.
// The handler is called for file writes, shell commands, URL fetches, and other operations.
func WithPermissionHandler(handler copilot.PermissionHandlerFunc) OptionFunc {
	return func(o *Options) { o.OnPermissionRequest = handler }
}
