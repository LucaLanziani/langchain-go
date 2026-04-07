package main

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 8: Custom Routing Logic
// Demonstrates using CustomStrategy to implement your own routing logic
// based on any criteria you choose.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Custom Strategy ===\n")

	// Create router with multiple providers
	router, err := provider.NewRouter(
		ctx,
		[]provider.ProviderEntry{
			{
				Name:         "openai-creative",
				ProviderType: provider.ProviderOpenAI,
				Options: []provider.ProviderOption{
					provider.WithModel("gpt-4o"),
					provider.WithTemperature(1.0), // High temperature for creativity
				},
			},
			{
				Name:         "anthropic-analytical",
				ProviderType: provider.ProviderAnthropic,
				Options: []provider.ProviderOption{
					provider.WithModel("claude-3-5-sonnet-20241022"),
					provider.WithMaxTokens(4096),
					provider.WithTemperature(0.3), // Low temperature for precision
				},
			},
			{
				Name:         "ollama-local",
				ProviderType: provider.ProviderOllama,
				Options: []provider.ProviderOption{
					provider.WithModel("llama3.2"),
					provider.WithBaseURL("http://localhost:11434"),
				},
			},
		},
		// CustomStrategy with user-defined routing logic
		&provider.CustomStrategy{
			SelectFunc: customRoutingLogic,
			OnSuccessFunc: func(ctx context.Context, providerName string, latency time.Duration) {
				fmt.Printf("  ✓ Success: %s (latency: %v)\n", providerName, latency)
			},
			OnErrorFunc: func(ctx context.Context, providerName string, err error) {
				fmt.Printf("  ✗ Error: %s - %v\n", providerName, err)
			},
		},
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with custom routing logic:")
	fmt.Println("  - Creative tasks → openai-creative (high temperature)")
	fmt.Println("  - Analytical tasks → anthropic-analytical (low temperature)")
	fmt.Println("  - Privacy-sensitive → ollama-local")
	fmt.Println("  - Code-related → anthropic-analytical")
	fmt.Println()

	// Test different types of requests

	// Test 1: Creative writing
	fmt.Println("=== Test 1: Creative Writing ===")
	fmt.Println("Request: Write a short poem about coding")
	response, err := router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Write a short poem about coding"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 2: Code analysis
	fmt.Println("=== Test 2: Code Analysis ===")
	fmt.Println("Request: Explain how binary search works")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Explain how binary search works with code examples"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 3: Mathematical analysis
	fmt.Println("=== Test 3: Mathematical Analysis ===")
	fmt.Println("Request: Calculate the derivative of x^2 + 3x + 5")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Calculate the derivative of x^2 + 3x + 5 and explain the steps"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 4: Privacy-sensitive request
	fmt.Println("=== Test 4: Privacy-Sensitive Request ===")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Summarize this confidential document"),
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
		fmt.Println()
	}

	fmt.Println("=== Example Complete ===")
}

// customRoutingLogic implements custom routing based on request content
func customRoutingLogic(ctx context.Context, reqCtx provider.RequestContext, providers map[string]llms.ChatModel) (string, error) {
	// Extract message content for analysis
	var content string
	if len(reqCtx.Messages) > 0 {
		content = strings.ToLower(reqCtx.Messages[len(reqCtx.Messages)-1].GetContent())
	}

	// Check for privacy-sensitive keywords
	privacyKeywords := []string{"confidential", "private", "secret", "sensitive"}
	for _, keyword := range privacyKeywords {
		if strings.Contains(content, keyword) {
			if _, exists := providers["ollama-local"]; exists {
				return "ollama-local", nil
			}
		}
	}

	// Route code-related requests to Anthropic (good at code)
	codeKeywords := []string{"code", "function", "algorithm", "programming", "debug", "implement"}
	for _, keyword := range codeKeywords {
		if strings.Contains(content, keyword) {
			if _, exists := providers["anthropic-analytical"]; exists {
				return "anthropic-analytical", nil
			}
		}
	}

	// Route creative requests to OpenAI with high temperature
	creativeKeywords := []string{"write", "story", "poem", "creative", "imagine", "brainstorm"}
	for _, keyword := range creativeKeywords {
		if strings.Contains(content, keyword) {
			if _, exists := providers["openai-creative"]; exists {
				return "openai-creative", nil
			}
		}
	}

	// Route analytical/mathematical requests to Anthropic
	analyticalKeywords := []string{"calculate", "analyze", "explain", "derive", "prove", "solve"}
	for _, keyword := range analyticalKeywords {
		if strings.Contains(content, keyword) {
			if _, exists := providers["anthropic-analytical"]; exists {
				return "anthropic-analytical", nil
			}
		}
	}

	// Default: use first available provider
	for name := range providers {
		return name, nil
	}

	return "", fmt.Errorf("no providers available")
}
