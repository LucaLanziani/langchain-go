// Package codex provides a chat model backed by an existing OpenAI Codex CLI
// subscription. It reads ~/.codex/auth.json for ChatGPT OAuth tokens (refreshing
// them when needed) and talks to ChatGPT's Codex backend using the Responses API.
package codex

// DefaultBaseURL is the Codex backend exposed to ChatGPT subscribers.
const DefaultBaseURL = "https://chatgpt.com/backend-api/codex"

// clientVersion is sent as the `client_version` query param to the backend.
// The /models endpoint rejects requests that omit it. Keep the value loosely
// in step with a recent codex CLI release.
const clientVersion = "0.99.0"

// DefaultModel is the model used by the Codex CLI when nothing is configured.
// The server may transparently route this to newer revisions (gpt-5.1-codex-*).
const DefaultModel = "gpt-5-codex"

// Options holds configuration for the Codex chat model.
type Options struct {
	// AuthFile overrides the path to auth.json. Empty uses $CODEX_HOME/auth.json
	// (fallback: ~/.codex/auth.json).
	AuthFile string

	// Model is the model ID to send to the Responses API (e.g. "gpt-5-codex").
	Model string

	// BaseURL overrides the Codex backend base URL.
	BaseURL string

	// Originator is sent as the "originator" header. Defaults to "codex_cli_rs"
	// so traffic looks like a regular Codex CLI session to the backend.
	Originator string

	// UserAgent overrides the User-Agent header.
	UserAgent string

	// ReasoningEffort controls reasoning depth for gpt-5-codex-family models.
	// Valid values: "minimal", "low", "medium", "high", "xhigh".
	ReasoningEffort string

	// ReasoningSummary controls the reasoning summary mode (e.g. "auto", "none").
	ReasoningSummary string

	// Instructions is appended to or used as the system-level instructions.
	// Most callers should pass a system message instead; this is here for
	// providers/codex consumers that want a baseline instruction string.
	Instructions string

	// MaxOutputTokens optionally caps response length.
	MaxOutputTokens *int

	// PromptCacheKey enables server-side caching of repeated prefixes.
	PromptCacheKey string

	// HTTPTimeout for non-streaming requests. Streaming uses no timeout.
	HTTPTimeout int // seconds; 0 means no timeout
}

// DefaultOptions returns sensible defaults for the Codex provider.
func DefaultOptions() *Options {
	return &Options{
		Model:            DefaultModel,
		BaseURL:          DefaultBaseURL,
		Originator:       "codex_cli_rs",
		UserAgent:        "codex_cli_rs",
		ReasoningSummary: "auto",
	}
}

// OptionFunc configures the Codex chat model.
type OptionFunc func(*Options)

// WithAuthFile overrides the path to auth.json.
func WithAuthFile(path string) OptionFunc {
	return func(o *Options) { o.AuthFile = path }
}

// WithModelName sets the model name (e.g. "gpt-5-codex").
func WithModelName(model string) OptionFunc {
	return func(o *Options) { o.Model = model }
}

// WithBaseURL overrides the Codex backend base URL.
func WithBaseURL(url string) OptionFunc {
	return func(o *Options) { o.BaseURL = url }
}

// WithReasoningEffort sets the reasoning effort for gpt-5-codex models.
func WithReasoningEffort(effort string) OptionFunc {
	return func(o *Options) { o.ReasoningEffort = effort }
}

// WithReasoningSummary sets the reasoning summary mode.
func WithReasoningSummary(summary string) OptionFunc {
	return func(o *Options) { o.ReasoningSummary = summary }
}

// WithInstructions sets a baseline instructions string for the Responses API.
func WithInstructions(instructions string) OptionFunc {
	return func(o *Options) { o.Instructions = instructions }
}

// WithMaxOutputTokens caps the output length.
func WithMaxOutputTokens(n int) OptionFunc {
	return func(o *Options) { o.MaxOutputTokens = &n }
}

// WithPromptCacheKey enables server-side prompt caching for a stable session.
func WithPromptCacheKey(key string) OptionFunc {
	return func(o *Options) { o.PromptCacheKey = key }
}

// WithOriginator overrides the originator header.
func WithOriginator(originator string) OptionFunc {
	return func(o *Options) { o.Originator = originator }
}

// WithUserAgent overrides the User-Agent header.
func WithUserAgent(ua string) OptionFunc {
	return func(o *Options) { o.UserAgent = ua }
}
