package llms

import "github.com/LucaLanziani/langchain-go/core"

// Common ChatModel option keys used in RunnableConfig.Configurable.
const (
	ConfigKeyTemperature = "temperature"
	ConfigKeyMaxTokens   = "max_tokens"
	ConfigKeyTopP        = "top_p"
	ConfigKeyModel       = "model"
	ConfigKeyResponseFmt = "response_format"
)

// WithTemperature sets the temperature for generation.
func WithTemperature(temp float64) core.Option {
	return core.WithConfigurable(map[string]any{ConfigKeyTemperature: temp})
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) core.Option {
	return core.WithConfigurable(map[string]any{ConfigKeyMaxTokens: n})
}

// WithTopP sets the top-p (nucleus sampling) parameter.
func WithTopP(p float64) core.Option {
	return core.WithConfigurable(map[string]any{ConfigKeyTopP: p})
}

// WithModel sets the model name.
func WithModel(model string) core.Option {
	return core.WithConfigurable(map[string]any{ConfigKeyModel: model})
}
