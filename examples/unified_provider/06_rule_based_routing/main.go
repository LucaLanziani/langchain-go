package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/provider"
)

// Example 6: Rule-Based Routing
// Demonstrates using RuleBasedStrategy to route requests based on
// request characteristics like complexity, token count, and custom metadata.

func main() {
	ctx := context.Background()

	fmt.Println("=== Creating Router with Rule-Based Strategy ===")

	// Create router with three providers
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
		// RuleBasedStrategy routes based on request characteristics
		provider.NewRuleBasedStrategy(
			[]provider.RoutingRule{
				{
					Name:     "complex-to-anthropic",
					Priority: 100,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Route complex requests to Anthropic
						return reqCtx.Complexity == "complex" || reqCtx.TotalTokens > 5000
					},
					Provider: "smart-anthropic",
				},
				{
					Name:     "simple-to-local",
					Priority: 90,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Route simple, short requests to the local LM Studio model
						return reqCtx.Complexity == "simple" && reqCtx.TotalTokens < 500
					},
					Provider: "local-lmstudio",
				},
				{
					Name:     "high-priority-to-openai",
					Priority: 80,
					Condition: func(reqCtx provider.RequestContext) bool {
						// Route high-priority requests to OpenAI
						return reqCtx.Priority == "high"
					},
					Provider: "fast-openai",
				},
			},
			"fast-openai", // default provider when no rule matches
		),
	)
	if err != nil {
		log.Fatalf("Failed to create router: %v", err)
	}
	defer router.Cleanup()

	fmt.Println("Router created with rule-based routing:")
	fmt.Println("  Rule 1 (Priority 100): Complex requests → smart-anthropic")
	fmt.Println("  Rule 2 (Priority 90): Simple requests → local-lmstudio")
	fmt.Println("  Rule 3 (Priority 80): High priority → fast-openai")
	fmt.Println("  Default: fast-openai")
	fmt.Println()

	// Test different types of requests

	// Test 1: Simple request (should go to local-lmstudio)
	fmt.Println("=== Test 1: Simple Request ===")
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

	// Test 2: Complex request (should go to smart-anthropic)
	fmt.Println("=== Test 2: Complex Request ===")
	complexQuestion := `Explain the differences between monolithic and microservices architectures, 
including their trade-offs, when to use each, and provide examples of companies using each approach. 
Also discuss the role of containers and orchestration in microservices.`
	fmt.Printf("Question: %s\n", complexQuestion)
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage(complexQuestion),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 3: High priority request (should go to fast-openai)
	fmt.Println("=== Test 3: Moderate Request ===")
	fmt.Println("Question: What is the capital of France?")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("What is the capital of France?"),
	})
	if err != nil {
		log.Printf("Request failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.GetContent())
	}
	fmt.Println()

	// Test 4: Default routing (moderate complexity, no special priority)
	fmt.Println("=== Test 4: Default Routing ===")
	fmt.Println("Question: Name three programming languages.")
	response, err = router.Invoke(ctx, []core.Message{
		core.NewHumanMessage("Name three programming languages."),
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
}
