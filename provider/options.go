package provider

import "github.com/LucaLanziani/langchain-go/tools"

// ===== Anthropic-Specific Options =====

// WithAnthropicVersion sets the Anthropic API version
func WithAnthropicVersion(version string) ProviderOption {
	return WithProviderSpecific("anthropic_version", version)
}

// ===== OpenAI-Specific Options =====

// WithOrganization sets the OpenAI organization ID
func WithOrganization(organization string) ProviderOption {
	return WithProviderSpecific("organization", organization)
}

// ===== Ollama-Specific Options =====

// WithKeepAlive sets the keep_alive duration for Ollama models
// Controls how long the model stays loaded in memory
// Examples: "5m", "10m", "-1" (keep forever), "0" (unload immediately)
func WithKeepAlive(keepAlive string) ProviderOption {
	return WithProviderSpecific("keep_alive", keepAlive)
}

// WithFormat sets the response format for Ollama
// Currently supports "json" for JSON-formatted responses
func WithFormat(format string) ProviderOption {
	return WithProviderSpecific("format", format)
}

// WithNumCtx sets the context window size for Ollama models
// Determines how many tokens the model can consider
func WithNumCtx(numCtx int) ProviderOption {
	return WithProviderSpecific("num_ctx", numCtx)
}

// WithTopK sets the top-k sampling parameter for Ollama
// Limits the next token selection to the K most likely tokens
func WithTopK(topK int) ProviderOption {
	return WithProviderSpecific("top_k", topK)
}

// ===== GitHub Copilot-Specific Options =====

// WithTools sets the tools available to GitHub Copilot
func WithTools(tools ...tools.Tool) ProviderOption {
	return WithProviderSpecific("tools", tools)
}

// WithCLIPath sets the path to the GitHub Copilot CLI executable
func WithCLIPath(cliPath string) ProviderOption {
	return WithProviderSpecific("cli_path", cliPath)
}

// WithLogLevel sets the log level for GitHub Copilot CLI
// Valid values: "debug", "info", "warn", "error"
func WithLogLevel(logLevel string) ProviderOption {
	return WithProviderSpecific("log_level", logLevel)
}
