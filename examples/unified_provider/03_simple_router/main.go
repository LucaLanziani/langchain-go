package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 3: Simple Router
// Demonstrates creating a router with multiple providers and using SimpleStrategy
// to always route to a specific provider.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Multiple Providers ===\n")

	// Create router with three providers
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "fast-openai",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o-mini"),
					provider.WithTemperature(0.7),
				},
			},
			{
				Name:         "smart-anthropic",
				ProviderType: provider.ProviderAnthropic,
				Options: []provider.ProviderOption{
					provider.WithModel("claude-3-5-haiku-20241022"),
					provider.WithMaxTokens(1000),
				},
			},
			{
				Name:         "local-ollama",
				ProviderType: provider.ProviderOllama,
				Options: []provider.ProviderOption{
					provider.WithModel("llama3.2"),
					provider.WithBaseURL("http://localhost:11434"),
				},
			},
		},
		// SimpleStrategy always routes to the specified provider
		&provider.SimpleStrategy{ProviderName: "fast-openai"},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with 3 providers:")
	fmt.Println("  - fast-openai (gpt-4o-mini)")
	fmt.Println("  - smart-anthropic (claude-3-5-haiku)")
	fmt.Println("  - local-ollama (llama3.2)")
	fmt.Println("\nUsing SimpleStrategy to always route to 'fast-openai'\n")

	// Make several requests - all will go to fast-openai
	questions := []string{
		"What is 2+2?",
		"What is the capital of Japan?",
		"Name a programming language.",
	}

	for i, question := range questions {
		fmt.Printf("Request %d: %s\n", i+1, question)

		response, err := router.Invoke(ctx, []core.Message{
			core.NewHumanMessage(question),
		})
		if err != nil {
			log.Printf("Request failed: %v\n", err)
			continue
		}

		fmt.Printf("Response: %s\n\n", response.GetContent())
	}

	// Display metrics
	fmt.Println("=== Router Metrics ===")
	metrics := router.GetMetrics()
	for name, m := range metrics {
		fmt.Printf("%s:\n", name)
		fmt.Printf("  Requests: %d\n", m.RequestCount)
		fmt.Printf("  Errors: %d\n", m.ErrorCount)
		if m.RequestCount > 0 {
			fmt.Printf("  Avg Latency: %v\n", m.TotalLatency/time.Duration(m.RequestCount))
		}
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
	fmt.Println("Note: All requests went to 'fast-openai' as configured by SimpleStrategy")
}
