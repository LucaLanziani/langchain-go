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
