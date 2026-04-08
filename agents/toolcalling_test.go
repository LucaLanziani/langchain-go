package agents

import (
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestFormatToolCallingStepsGeneratesUniqueIDs(t *testing.T) {
	steps := []AgentStep{
		{
			Action:      AgentAction{Tool: "search", ToolInput: `{"query":"golang"}`},
			Observation: "first",
		},
		{
			Action:      AgentAction{Tool: "search", ToolInput: `{"query":"langchain"}`},
			Observation: "second",
		},
	}

	messages := formatToolCallingSteps(steps)
	if len(messages) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(messages))
	}

	firstAI := messages[0].(*core.AIMessage)
	firstTool := messages[1].(*core.ToolMessage)
	secondAI := messages[2].(*core.AIMessage)
	secondTool := messages[3].(*core.ToolMessage)

	if firstAI.ToolCalls[0].ID == secondAI.ToolCalls[0].ID {
		t.Fatalf("expected unique tool call IDs, got %q", firstAI.ToolCalls[0].ID)
	}
	if firstTool.ToolCallID != firstAI.ToolCalls[0].ID {
		t.Fatalf("expected first tool message to use %q, got %q", firstAI.ToolCalls[0].ID, firstTool.ToolCallID)
	}
	if secondTool.ToolCallID != secondAI.ToolCalls[0].ID {
		t.Fatalf("expected second tool message to use %q, got %q", secondAI.ToolCalls[0].ID, secondTool.ToolCallID)
	}
}

func TestFormatToolCallingStepsPreservesToolCallID(t *testing.T) {
	steps := []AgentStep{
		{
			Action:      AgentAction{Tool: "search", ToolInput: `{"query":"golang"}`, ToolCallID: "call_123"},
			Observation: "done",
		},
	}

	messages := formatToolCallingSteps(steps)
	aiMsg := messages[0].(*core.AIMessage)
	toolMsg := messages[1].(*core.ToolMessage)

	if aiMsg.ToolCalls[0].ID != "call_123" {
		t.Fatalf("expected preserved tool call ID, got %q", aiMsg.ToolCalls[0].ID)
	}
	if toolMsg.ToolCallID != "call_123" {
		t.Fatalf("expected preserved tool message ID, got %q", toolMsg.ToolCallID)
	}
}
