// Example: GitHub Copilot chat — demonstrates using the Copilot SDK provider.
//
// This creates a Copilot chat model and shows basic invocation, streaming,
// and tool use with the SDK-managed tool loop.
//
// Prerequisites:
//   - Copilot CLI installed (https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli)
//   - GitHub authentication via copilot CLI login or GITHUB_TOKEN env var
//
// Usage:
//
//	go run ./examples/copilot_chat/
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
	copilot "github.com/LucaLanziani/langchain-go/providers/github-copilot"
	"github.com/LucaLanziani/langchain-go/tools"
)

func main() {
	ctx := context.Background()

	// Define a simple tool that the SDK will manage automatically.
	calculator := tools.NewTool(
		"calculator",
		"Useful for math calculations. Input should be a math expression.",
		func(_ context.Context, input string) (string, error) {
			return "42", nil
		},
	)

	// Create the model with the default gpt-5-mini and a tool.
	model, err := copilot.New(ctx,
		copilot.WithTools(calculator),
	)
	if err != nil {
		log.Fatalf("Failed to create Copilot model: %v", err)
	}
	defer model.Close()

	// --- Basic invocation ---
	fmt.Println("=== Basic Invocation ===")
	response, err := model.Invoke(ctx, []core.Message{
		core.NewSystemMessage("You are a helpful assistant. Be concise."),
		core.NewHumanMessage("What is Go programming language in one sentence?"),
	})
	if err != nil {
		log.Fatalf("Invoke error: %v", err)
	}
	fmt.Println(response.Content)
	fmt.Println()

	// --- Streaming ---
	fmt.Println("=== Streaming ===")
	stream, err := model.Stream(ctx, []core.Message{
		core.NewSystemMessage("You are a helpful assistant. Be concise."),
		core.NewHumanMessage("Tell me a very short joke."),
	})
	if err != nil {
		log.Fatalf("Stream error: %v", err)
	}
	for {
		chunk, ok, err := stream.Next()
		if err != nil {
			log.Fatalf("Stream chunk error: %v", err)
		}
		if !ok {
			break
		}
		fmt.Print(chunk.Content)
	}
	fmt.Println()
	fmt.Println()

	// --- Tool use (SDK-managed loop) ---
	fmt.Println("=== Tool Use (SDK-managed) ===")
	toolResponse, err := model.Invoke(ctx, []core.Message{
		core.NewSystemMessage("You are a helpful assistant. Use the calculator tool for math."),
		core.NewHumanMessage("What is 6 * 7?"),
	})
	if err != nil {
		log.Fatalf("Tool invoke error: %v", err)
	}
	fmt.Println(toolResponse.Content)
	fmt.Println()

	// --- Batch (parallel) ---
	fmt.Println("=== Batch (Parallel) ===")
	batchResults, err := model.Batch(ctx, [][]core.Message{
		{core.NewHumanMessage("Say hello in Spanish")},
		{core.NewHumanMessage("Say hello in French")},
		{core.NewHumanMessage("Say hello in Japanese")},
	})
	if err != nil {
		log.Fatalf("Batch error: %v", err)
	}
	for i, r := range batchResults {
		fmt.Printf("  %d: %s\n", i+1, r.Content)
	}

	fmt.Println()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("Done!")
}
