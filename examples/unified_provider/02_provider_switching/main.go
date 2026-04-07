package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 2: Provider Switching
// Demonstrates how easy it is to switch between providers using the unified interface.
// The same code works with any provider - just change the provider type.

func main() {
	ctx := context.Background()

	// Get provider from environment or default to OpenAI
	providerName := os.Getenv("LLM_PROVIDER")
	if providerName == "" {
		providerName = "openai"
	}

	fmt.Printf("=== Using Provider: %s ===\n\n", providerName)

	// Create model using the selected provider
	model, cleanup, err := createModel(ctx, providerName)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	defer cleanup()

	// Use the model - same code works for all providers
	messages := []core.Message{
		core.NewHumanMessage("What is the capital of France? Answer in one sentence."),
	}

	response, err := model.Invoke(ctx, messages)
	if err != nil {
		log.Fatalf("Invoke failed: %v", err)
	}

	fmt.Printf("Response: %s\n\n", response.GetContent())

	// Try a follow-up question
	messages = append(messages, response)
	messages = append(messages, core.NewHumanMessage("What is its population?"))

	response, err = model.Invoke(ctx, messages)
	if err != nil {
		log.Fatalf("Invoke failed: %v", err)
	}

	fmt.Printf("Follow-up Response: %s\n\n", response.GetContent())
	fmt.Println("=== Example Complete ===")
	fmt.Println("\nTry switching providers:")
	fmt.Println("  LLM_PROVIDER=openai go run main.go")
	fmt.Println("  LLM_PROVIDER=anthropic go run main.go")
	fmt.Println("  LLM_PROVIDER=ollama go run main.go")
	fmt.Println("  LLM_PROVIDER=copilot go run main.go")
}

// createModel creates a provider based on the provider name
// This demonstrates how the unified interface makes provider switching trivial
func createModel(ctx context.Context, providerName string) (llms.ChatModel, provider.CleanupFunc, error) {
	var providerType provider.ProviderType
	var opts []provider.ProviderOption

	switch providerName {
	case "openai":
		providerType = provider.ProviderOpenAI
		opts = []provider.ProviderOption{
			provider.WithModel("gpt-4o-mini"),
			provider.WithTemperature(0.7),
			provider.WithMaxTokens(200),
		}

	case "anthropic":
		providerType = provider.ProviderAnthropic
		opts = []provider.ProviderOption{
			provider.WithModel("claude-3-5-haiku-20241022"),
			provider.WithTemperature(0.7),
			provider.WithMaxTokens(200),
		}

	case "ollama":
		providerType = provider.ProviderOllama
		opts = []provider.ProviderOption{
			provider.WithModel("llama3.2"),
			provider.WithBaseURL("http://localhost:11434"),
			provider.WithTemperature(0.7),
		}

	case "copilot":
		providerType = provider.ProviderGitHubCopilot
		opts = []provider.ProviderOption{
			provider.WithModel("gpt-4o"),
			provider.WithTemperature(0.7),
		}

	default:
		return nil, nil, fmt.Errorf("unknown provider: %s", providerName)
	}

	return provider.NewProvider(ctx, providerType, opts...)
}
