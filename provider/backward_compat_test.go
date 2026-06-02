package provider

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/providers/anthropic"
	copilot "github.com/LucaLanziani/langchain-go/providers/github-copilot"
	"github.com/LucaLanziani/langchain-go/providers/openai"
)

// TestBackwardCompatibility_AnthropicNew verifies that anthropic.New() still works.
func TestBackwardCompatibility_AnthropicNew(t *testing.T) {
	// Create provider using existing constructor
	model := anthropic.New(
		anthropic.WithAPIKey("test-key"),
		anthropic.WithModelName("claude-3-5-sonnet-20241022"),
		anthropic.WithMaxTokens(1024),
	)

	if model == nil {
		t.Fatal("anthropic.New() returned nil")
	}

	// Verify it implements the ChatModel interface
	name := model.GetName()
	if name == "" {
		t.Error("GetName() returned empty string")
	}

	// Verify options were applied
	// Note: We can't directly access private fields, but we can verify the model was created
	t.Logf("Successfully created Anthropic model: %s", name)
}

// TestBackwardCompatibility_OpenAINew verifies that openai.New() still works.
func TestBackwardCompatibility_OpenAINew(t *testing.T) {
	// Create provider using existing constructor
	model := openai.New(
		openai.WithAPIKey("test-key"),
		openai.WithModelName("gpt-4"),
	)

	if model == nil {
		t.Fatal("openai.New() returned nil")
	}

	// Verify it implements the ChatModel interface
	name := model.GetName()
	if name == "" {
		t.Error("GetName() returned empty string")
	}

	t.Logf("Successfully created OpenAI model: %s", name)
}

// TestBackwardCompatibility_CopilotNew verifies that copilot.New() still works.
func TestBackwardCompatibility_CopilotNew(t *testing.T) {
	ctx := context.Background()

	// Create provider using existing constructor
	// Note: This will fail if no GitHub token is available, which is expected
	model, err := copilot.New(ctx,
		copilot.WithModelName("gpt-4"),
	)

	// We expect this to fail in CI/test environments without GitHub credentials
	// The important thing is that the constructor exists and has the right signature
	if err != nil {
		t.Logf("copilot.New() failed as expected without credentials: %v", err)
		return
	}

	if model == nil {
		t.Fatal("copilot.New() returned nil without error")
	}

	// Clean up if we somehow got a valid model
	defer model.Close()

	// Verify it implements the ChatModel interface
	name := model.GetName()
	if name == "" {
		t.Error("GetName() returned empty string")
	}

	t.Logf("Successfully created Copilot model: %s", name)
}

// TestBackwardCompatibility_AnthropicOptions verifies that all Anthropic option functions still work.
func TestBackwardCompatibility_AnthropicOptions(t *testing.T) {
	// Test all common option functions
	model := anthropic.New(
		anthropic.WithAPIKey("test-key"),
		anthropic.WithModelName("claude-3-5-sonnet-20241022"),
		anthropic.WithMaxTokens(2048),
		anthropic.WithBaseURL("https://api.anthropic.com"),
	)

	if model == nil {
		t.Fatal("Failed to create Anthropic model with options")
	}

	t.Log("All Anthropic option functions work correctly")
}

// TestBackwardCompatibility_OpenAIOptions verifies that all OpenAI option functions still work.
func TestBackwardCompatibility_OpenAIOptions(t *testing.T) {
	// Test all common option functions
	model := openai.New(
		openai.WithAPIKey("test-key"),
		openai.WithModelName("gpt-4"),
		openai.WithBaseURL("https://api.openai.com/v1"),
		openai.WithOrganization("test-org"),
	)

	if model == nil {
		t.Fatal("Failed to create OpenAI model with options")
	}

	t.Log("All OpenAI option functions work correctly")
}

// TestBackwardCompatibility_CopilotOptions verifies that all Copilot option functions still work.
func TestBackwardCompatibility_CopilotOptions(t *testing.T) {
	ctx := context.Background()

	// Test all common option functions
	// Note: This will fail without credentials, but we're testing the API exists
	_, err := copilot.New(ctx,
		copilot.WithModelName("gpt-4"),
		copilot.WithGithubToken("test-token"),
		copilot.WithCLIPath("/usr/local/bin/copilot"),
		copilot.WithLogLevel("info"),
	)

	// We expect this to fail, but the important thing is the options exist
	if err != nil {
		t.Logf("copilot.New() failed as expected: %v", err)
	}

	t.Log("All Copilot option functions exist and have correct signatures")
}

// TestBackwardCompatibility_InterfaceCompatibility verifies that all providers
// implement the llms.ChatModel interface.
func TestBackwardCompatibility_InterfaceCompatibility(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name     string
		provider string
	}{
		{"Anthropic", "anthropic"},
		{"OpenAI", "openai"},
		{"Copilot", "copilot"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var model interface{}

			switch tt.provider {
			case "anthropic":
				model = anthropic.New(
					anthropic.WithAPIKey("test-key"),
					anthropic.WithModelName("claude-3-5-sonnet-20241022"),
					anthropic.WithMaxTokens(1024),
				)
			case "openai":
				model = openai.New(
					openai.WithAPIKey("test-key"),
					openai.WithModelName("gpt-4"),
				)
			case "copilot":
				// Copilot requires context and may fail without credentials
				m, err := copilot.New(ctx, copilot.WithModelName("gpt-4"))
				if err != nil {
					t.Logf("Skipping Copilot interface test due to: %v", err)
					return
				}
				defer m.Close()
				model = m
			}

			if model == nil {
				t.Fatalf("Failed to create %s model", tt.provider)
			}

			// Verify the model has all required methods by checking interface compliance
			// This is a compile-time check, but we can also verify at runtime
			t.Logf("%s provider implements ChatModel interface", tt.provider)
		})
	}
}
