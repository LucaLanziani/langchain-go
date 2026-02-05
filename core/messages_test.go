package core

import (
	"testing"
)

func TestHumanMessage(t *testing.T) {
	msg := NewHumanMessage("hello")
	if msg.GetType() != MessageTypeHuman {
		t.Errorf("expected %s, got %s", MessageTypeHuman, msg.GetType())
	}
	if msg.GetContent() != "hello" {
		t.Errorf("expected 'hello', got %q", msg.GetContent())
	}
}

func TestAIMessage(t *testing.T) {
	msg := NewAIMessage("response")
	if msg.GetType() != MessageTypeAI {
		t.Errorf("expected %s, got %s", MessageTypeAI, msg.GetType())
	}
	if msg.GetContent() != "response" {
		t.Errorf("expected 'response', got %q", msg.GetContent())
	}
}

func TestAIMessageWithToolCalls(t *testing.T) {
	toolCalls := []ToolCall{
		{ID: "call_1", Name: "search", Args: []byte(`{"query":"test"}`)},
	}
	msg := NewAIMessageWithToolCalls("thinking...", toolCalls)
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].Name != "search" {
		t.Errorf("expected tool call name 'search', got %q", msg.ToolCalls[0].Name)
	}
	if msg.GetContent() != "thinking..." {
		t.Errorf("expected 'thinking...', got %q", msg.GetContent())
	}
}

func TestSystemMessage(t *testing.T) {
	msg := NewSystemMessage("You are helpful")
	if msg.GetType() != MessageTypeSystem {
		t.Errorf("expected %s, got %s", MessageTypeSystem, msg.GetType())
	}
}

func TestToolMessage(t *testing.T) {
	msg := NewToolMessage("result", "call_123")
	if msg.GetType() != MessageTypeTool {
		t.Errorf("expected %s, got %s", MessageTypeTool, msg.GetType())
	}
	if msg.ToolCallID != "call_123" {
		t.Errorf("expected ToolCallID 'call_123', got %q", msg.ToolCallID)
	}
}

func TestFunctionMessage(t *testing.T) {
	msg := NewFunctionMessage("my_func", "result")
	if msg.GetType() != MessageTypeFunction {
		t.Errorf("expected %s, got %s", MessageTypeFunction, msg.GetType())
	}
	if msg.GetName() != "my_func" {
		t.Errorf("expected name 'my_func', got %q", msg.GetName())
	}
}

func TestGenericMessage(t *testing.T) {
	msg := NewGenericMessage("custom_role", "hello")
	if msg.GetType() != MessageTypeGeneric {
		t.Errorf("expected %s, got %s", MessageTypeGeneric, msg.GetType())
	}
	if msg.Role != "custom_role" {
		t.Errorf("expected role 'custom_role', got %q", msg.Role)
	}
}

func TestGetBufferString(t *testing.T) {
	messages := []Message{
		NewHumanMessage("Hello"),
		NewAIMessage("Hi there!"),
		NewSystemMessage("Be helpful"),
	}
	result := GetBufferString(messages, "Human", "AI")
	expected := "Human: Hello\nAI: Hi there!\nSystem: Be helpful"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestGetBufferStringDefaults(t *testing.T) {
	messages := []Message{
		NewHumanMessage("test"),
	}
	result := GetBufferString(messages, "", "")
	if result != "Human: test" {
		t.Errorf("expected 'Human: test', got %q", result)
	}
}
