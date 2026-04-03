// Package anthropic provides an Anthropic/Claude chat model implementation.
package anthropic

// Options holds configuration for the Anthropic chat model.
type Options struct {
	// APIKey is the Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
	APIKey string

	// Model is the model ID (e.g., "claude-sonnet-4-20250514", "claude-3-haiku-20240307").
	Model string

	// BaseURL overrides the API base URL.
	BaseURL string

	// Temperature controls randomness (0.0 to 1.0).
	Temperature *float64

	// MaxTokens limits the response length. Required by Anthropic API.
	MaxTokens int

	// TopP controls nucleus sampling.
	TopP *float64

	// Stop sequences.
	Stop []string

	// ThinkingBudget, when > 0, enables Anthropic Extended Thinking and sets
	// the maximum number of tokens the model may use for its internal reasoning.
	// Requires a model that supports extended thinking (e.g. claude-3-7-sonnet-*).
	// When enabled the API enforces temperature=1 and disables top_p/top_k.
	ThinkingBudget int
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() *Options {
	return &Options{
		Model:     "claude-sonnet-4-20250514",
		BaseURL:   "https://api.anthropic.com/v1",
		MaxTokens: 4096,
	}
}

// OptionFunc configures Anthropic-specific options.
type OptionFunc func(*Options)

// WithAPIKey sets the API key.
func WithAPIKey(key string) OptionFunc {
	return func(o *Options) { o.APIKey = key }
}

// WithModelName sets the model name.
func WithModelName(model string) OptionFunc {
	return func(o *Options) { o.Model = model }
}

// WithBaseURL sets the API base URL.
func WithBaseURL(url string) OptionFunc {
	return func(o *Options) { o.BaseURL = url }
}

// WithMaxTokens sets the maximum tokens.
func WithMaxTokens(n int) OptionFunc {
	return func(o *Options) { o.MaxTokens = n }
}

// WithThinkingBudget enables Anthropic Extended Thinking with the given token budget.
// Set to 0 (default) to disable. The budget must be less than MaxTokens.
// When enabled, temperature is forced to 1 per the Anthropic API requirement.
func WithThinkingBudget(budget int) OptionFunc {
	return func(o *Options) { o.ThinkingBudget = budget }
}
