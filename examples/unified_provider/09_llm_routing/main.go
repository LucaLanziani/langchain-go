package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 9: LLM-Based Intelligent Routing
// NOTE: This example demonstrates the concept of LLM-based routing.
// The LLMRoutingStrategy has unexported fields, so this example shows
// how it would work conceptually. In practice, you would use rule-based
// or custom strategies to achieve similar intelligent routing.

func main() {
	ctx := context.Background()

	fmt.Println("=== LLM-Based Routing Concept ===")
	fmt.Println("This example demonstrates intelligent routing using a custom strategy")
	fmt.Println("that analyzes request characteristics to select the best provider.")
	fmt.Println()

	// Create router with multiple providers and intelligent custom routing
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "fast-openai",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-3.5-turbo"),
					provider.WithTemperature(0.7),
				},
			},
			{
				Name:         "smart-anthropic",
				ProviderType: provider.ProviderAnthropic,
				Options: []provider.ProviderOption{
					provider.WithModel("claude-3-5-sonnet-20241022"),
					provider.WithMaxTokens(4096),
				},
			},
			{
				Name:         "local-lmstudio",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("qwen2.5-7b-instruct"),
					provider.WithBaseURL("http://localhost:1234/v1"),
					provider.WithAPIKey("lm-studio"),
				},
			},
		},
		// Use custom strategy for intelligent routing
		&provider.CustomStrategy{
			SelectFunc: intelligentRoutingLogic,
		},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("\nRouter created with intelligent custom routing")
	fmt.Println("Routing logic:")
	fmt.Println("  - Simple queries → fast-openai")
	fmt.Println("  - Complex reasoning/code → smart-anthropic")
	fmt.Println("  - Privacy-sensitive → local-lmstudio")
	fmt.Println()

	// Test different types of requests

	// Test 1: Simple question (should route to fast-openai)
	fmt.Println("=== Test 1: Simple Question ===")
	fmt.Println("Question: What is 2+2?")
	response, err := router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is 2+2?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 2: Complex code request (should route to smart-anthropic)
	fmt.Println("=== Test 2: Complex Code Request ===")
	codeQuestion := "Write a concurrent web scraper in Go with rate limiting, error handling, and graceful shutdown"
	fmt.Printf("Question: %s\n", codeQuestion)
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage(codeQuestion),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 3: Privacy-sensitive request (should route to local-lmstudio)
	fmt.Println("=== Test 3: Privacy-Sensitive Request ===")
	fmt.Println("Question: Analyze this confidential business strategy")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Analyze this confidential business strategy document and provide insights"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 4: Another simple question
	fmt.Println("=== Test 4: Another Simple Question ===")
	fmt.Println("Question: What is 3+3?")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is 3+3?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 5: Complex analysis (should route to smart-anthropic)
	fmt.Println("=== Test 5: Complex Analysis ===")
	analysisQuestion := "Explain the trade-offs between microservices and monolithic architectures"
	fmt.Printf("Question: %s\n", analysisQuestion)
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage(analysisQuestion),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

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
	fmt.Println("Note: This example uses a custom strategy to demonstrate intelligent")
	fmt.Println("routing based on request characteristics. In a real LLM-based routing")
	fmt.Println("system, an LLM would analyze each request and make routing decisions,")
	fmt.Println("with caching to avoid repeated LLM calls for similar requests.")
}

// intelligentRoutingLogic implements intelligent routing based on request characteristics
// This demonstrates the concept of LLM-based routing using rule-based logic
func intelligentRoutingLogic(ctx context.Context, reqCtx provider.RequestContext, providers map[string]llms.ChatModel) (string, error) {
	// Route based on complexity and characteristics

	// Complex requests or those with tool calls → smart-anthropic
	if reqCtx.Complexity == "complex" || reqCtx.HasToolCalls || reqCtx.TotalTokens > 5000 {
		if _, exists := providers["smart-anthropic"]; exists {
			return "smart-anthropic", nil
		}
	}

	// Check for privacy-sensitive requests
	if privacy, ok := reqCtx.UserMetadata["privacy"].(string); ok && privacy == "high" {
		if _, exists := providers["local-lmstudio"]; exists {
			return "local-lmstudio", nil
		}
	}

	// Simple, short requests → fast-openai (default)
	if _, exists := providers["fast-openai"]; exists {
		return "fast-openai", nil
	}

	// Fallback to any available provider
	for name := range providers {
		return name, nil
	}

	return "", fmt.Errorf("no providers available")
}
