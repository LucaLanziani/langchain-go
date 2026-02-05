// Example: Simple chain using prompt + model + output parser.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/simple_chain/
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/langchain-go/langchain-go/outputparsers"
	"github.com/langchain-go/langchain-go/prompts"
	"github.com/langchain-go/langchain-go/providers/openai"
	"github.com/langchain-go/langchain-go/runnable"
)

func main() {
	ctx := context.Background()

	// Create the components.
	prompt := prompts.NewChatPromptTemplate(
		prompts.System("You are a helpful assistant that tells jokes."),
		prompts.Human("Tell me a short joke about {topic}"),
	)

	model := openai.New()
	parser := outputparsers.NewStringOutputParser()

	// Compose: prompt -> model -> parser
	chain := runnable.Pipe3(prompt, model, parser)

	// Run the chain.
	result, err := chain.Invoke(ctx, map[string]any{"topic": "golang"})
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Println(result)
}
