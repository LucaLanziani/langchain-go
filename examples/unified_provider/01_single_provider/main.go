package main

import (
	"context"
	"fmt"
	"log"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 1: Single Provider Creation
// Demonstrates creating individual providers (OpenAI, Anthropic, LM Studio, Copilot)
// using the unified provider interface.

func main() {
	ctx := context.Background()

	// Example 1a: Create OpenAI provider
	fmt.Println("=== Creating OpenAI Provider ===")
	openaiModel, openaiCleanup, err := provider.NewProvider(
		ctx,
		provider.ProviderOpenAI,
		provider.WithModel("gpt-4o-mini"),
		provider.WithTemperature(0.7),
		provider.WithMaxTokens(100),
	)
	if err != nil {
		log.Printf("Failed to create OpenAI provider: %v\n", err)
	} else {
		defer openaiCleanup()
		response, err := openaiModel.Invoke(ctx, []core.Message{
			core.NewHumanMessage("Say hello in one sentence"),
		})
		if err != nil {
			log.Printf("OpenAI invoke failed: %v\n", err)
		} else {
			fmt.Printf("OpenAI Response: %s\n\n", response.GetContent())
		}
	}

	// Example 1b: Create Anthropic provider
	fmt.Println("=== Creating Anthropic Provider ===")
	anthropicModel, anthropicCleanup, err := provider.NewProvider(
		ctx,
		provider.ProviderAnthropic,
		provider.WithModel("claude-3-5-haiku-20241022"),
		provider.WithTemperature(0.7),
		provider.WithMaxTokens(100), // Required for Anthropic
	)
	if err != nil {
		log.Printf("Failed to create Anthropic provider: %v\n", err)
	} else {
		defer anthropicCleanup()
		response, err := anthropicModel.Invoke(ctx, []core.Message{
			core.NewHumanMessage("Say hello in one sentence"),
		})
		if err != nil {
			log.Printf("Anthropic invoke failed: %v\n", err)
		} else {
			fmt.Printf("Anthropic Response: %s\n\n", response.GetContent())
		}
	}

	// Example 1c: Create LM Studio provider (local via OpenAI-compatible endpoint)
	fmt.Println("=== Creating LM Studio Provider ===")
	lmStudioModel, lmStudioCleanup, err := provider.NewProvider(
		ctx,
		provider.ProviderOpenAI,
		provider.WithModel("qwen2.5-7b-instruct"),
		provider.WithBaseURL("http://localhost:1234/v1"),
		provider.WithAPIKey("lm-studio"),
		provider.WithTemperature(0.7),
	)
	if err != nil {
		log.Printf("Failed to create LM Studio provider: %v\n", err)
	} else {
		defer lmStudioCleanup()
		response, err := lmStudioModel.Invoke(ctx, []core.Message{
			core.NewHumanMessage("Say hello in one sentence"),
		})
		if err != nil {
			log.Printf("LM Studio invoke failed: %v\n", err)
		} else {
			fmt.Printf("LM Studio Response: %s\n\n", response.GetContent())
		}
	}

	// Example 1d: Create GitHub Copilot provider
	fmt.Println("=== Creating GitHub Copilot Provider ===")
	copilotModel, copilotCleanup, err := provider.NewProvider(
		ctx,
		provider.ProviderGitHubCopilot,
		provider.WithModel("gpt-4o"),
		provider.WithTemperature(0.7),
	)
	if err != nil {
		log.Printf("Failed to create Copilot provider: %v\n", err)
	} else {
		defer copilotCleanup()
		response, err := copilotModel.Invoke(ctx, []core.Message{
			core.NewHumanMessage("Say hello in one sentence"),
		})
		if err != nil {
			log.Printf("Copilot invoke failed: %v\n", err)
		} else {
			fmt.Printf("Copilot Response: %s\n\n", response.GetContent())
		}
	}

	fmt.Println("=== Example Complete ===")
}
