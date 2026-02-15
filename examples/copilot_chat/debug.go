package main

import (
	"context"
	"fmt"
	"os"
	"time"

	copilotSDK "github.com/github/copilot-sdk/go"
)

func init() {
	// Run debug test when COPILOT_DEBUG=1
	if os.Getenv("COPILOT_DEBUG") != "1" {
		return
	}

	fmt.Println("DEBUG: Starting copilot SDK test...")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("DEBUG: Creating client...")
	client := copilotSDK.NewClient(&copilotSDK.ClientOptions{
		LogLevel: "debug",
	})

	fmt.Println("DEBUG: Starting client...")
	if err := client.Start(ctx); err != nil {
		fmt.Printf("DEBUG: Start failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("DEBUG: Client started successfully!")

	fmt.Println("DEBUG: Creating session...")
	session, err := client.CreateSession(ctx, &copilotSDK.SessionConfig{
		Model: "gpt-5-mini",
	})
	if err != nil {
		fmt.Printf("DEBUG: CreateSession failed: %v\n", err)
		client.Stop()
		os.Exit(1)
	}
	fmt.Println("DEBUG: Session created!")

	fmt.Println("DEBUG: Sending message...")
	resp, err := session.SendAndWait(ctx, copilotSDK.MessageOptions{
		Prompt: "Say hello",
	})
	if err != nil {
		fmt.Printf("DEBUG: SendAndWait failed: %v\n", err)
	} else if resp != nil && resp.Data.Content != nil {
		fmt.Printf("DEBUG: Response: %s\n", *resp.Data.Content)
	}

	client.Stop()
	fmt.Println("DEBUG: Done!")
	os.Exit(0)
}
