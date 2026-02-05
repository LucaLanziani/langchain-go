// Package openai provides an OpenAI chat model implementation.
package openai

// Options holds configuration for the OpenAI chat model.
type Options struct {
	// APIKey is the OpenAI API key. Falls back to OPENAI_API_KEY env var.
	APIKey string

	// Model is the model ID (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo").
	Model string

	// BaseURL overrides the API base URL. Useful for proxies or Azure.
	BaseURL string

	// Organization is the OpenAI organization ID.
	Organization string

	// Temperature controls randomness (0.0 to 2.0).
	Temperature *float64

	// MaxTokens limits the response length.
	MaxTokens *int

	// TopP controls nucleus sampling.
	TopP *float64

	// Stop sequences.
	Stop []string

	// ResponseFormat can be "text" or "json_object".
	ResponseFormat string
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() *Options {
	return &Options{
		Model:   "gpt-4o",
		BaseURL: "https://api.openai.com/v1",
	}
}

// OptionFunc configures OpenAI-specific options.
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

// WithOrganization sets the organization ID.
func WithOrganization(org string) OptionFunc {
	return func(o *Options) { o.Organization = org }
}
