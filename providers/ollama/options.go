// Package ollama provides a chat model and embeddings implementation using the Ollama HTTP API.
package ollama

// options holds configuration for the Ollama provider.
type options struct {
	// BaseURL is the Ollama server base URL. Defaults to "http://localhost:11434".
	BaseURL string

	// Model is the model name (e.g., "llama3.1", "mistral", "phi3"). Defaults to "llama3.1".
	Model string

	// Temperature controls randomness (0.0 to 1.0).
	Temperature *float64

	// TopP controls nucleus sampling.
	TopP *float64

	// TopK limits the token sampling pool.
	TopK *int

	// NumPredict is the maximum number of tokens to generate (-1 for unlimited).
	NumPredict *int

	// Stop is a list of stop sequences.
	Stop []string

	// NumCtx is the size of the context window (in tokens).
	NumCtx *int

	// Format forces the output format. Use "json" for JSON output.
	Format string

	// KeepAlive controls how long the model stays loaded in memory (e.g., "5m").
	KeepAlive string
}

// defaultOptions returns sensible defaults for the Ollama chat provider.
func defaultOptions() *options {
	return &options{
		BaseURL: "http://localhost:11434",
		Model:   "llama3.1",
	}
}

// defaultEmbeddingOptions returns sensible defaults for the Ollama embeddings provider.
func defaultEmbeddingOptions() *options {
	return &options{
		BaseURL: "http://localhost:11434",
		Model:   "nomic-embed-text",
	}
}

// OptionFunc configures Ollama provider options.
type OptionFunc func(*options)

// WithModel sets the model name.
func WithModel(model string) OptionFunc {
	return func(o *options) { o.Model = model }
}

// WithBaseURL sets the Ollama server base URL.
func WithBaseURL(url string) OptionFunc {
	return func(o *options) { o.BaseURL = url }
}

// WithTemperature sets the sampling temperature.
func WithTemperature(temp float64) OptionFunc {
	return func(o *options) { o.Temperature = &temp }
}

// WithTopP sets the nucleus sampling parameter.
func WithTopP(p float64) OptionFunc {
	return func(o *options) { o.TopP = &p }
}

// WithTopK sets the top-k sampling parameter.
func WithTopK(k int) OptionFunc {
	return func(o *options) { o.TopK = &k }
}

// WithNumPredict sets the maximum number of tokens to generate.
func WithNumPredict(n int) OptionFunc {
	return func(o *options) { o.NumPredict = &n }
}

// WithStop sets the stop sequences.
func WithStop(stop []string) OptionFunc {
	return func(o *options) { o.Stop = stop }
}

// WithNumCtx sets the context window size in tokens.
func WithNumCtx(n int) OptionFunc {
	return func(o *options) { o.NumCtx = &n }
}

// WithFormat sets the output format ("json" or "").
func WithFormat(format string) OptionFunc {
	return func(o *options) { o.Format = format }
}

// WithKeepAlive sets how long the model stays loaded in memory (e.g., "5m").
func WithKeepAlive(d string) OptionFunc {
	return func(o *options) { o.KeepAlive = d }
}
