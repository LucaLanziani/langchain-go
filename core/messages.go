// Package core provides the fundamental types and interfaces for langchain-go.
package core

import (
	"encoding/json"
)

// MessageType identifies the role/type of a message.
type MessageType string

const (
	MessageTypeHuman    MessageType = "human"
	MessageTypeAI       MessageType = "ai"
	MessageTypeSystem   MessageType = "system"
	MessageTypeTool     MessageType = "tool"
	MessageTypeFunction MessageType = "function"
	MessageTypeGeneric  MessageType = "generic"
)

// ToolCall represents a request from the AI to invoke a tool.
type ToolCall struct {
	ID       string          `json:"id"`
	Name     string          `json:"name"`
	Args     json.RawMessage `json:"args"`
	Type     string          `json:"type,omitempty"`
}

// ToolCallChunk represents a streaming chunk of a tool call.
type ToolCallChunk struct {
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Args  string `json:"args,omitempty"`
	Index int    `json:"index,omitempty"`
}

// ContentBlock represents a block of content within a message.
// It can be text, an image, or other content types.
type ContentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// Message is the interface all message types implement.
type Message interface {
	// GetType returns the message type (human, ai, system, tool, function).
	GetType() MessageType
	// GetContent returns the text content of the message.
	GetContent() string
	// GetName returns the optional name associated with the message.
	GetName() string
	// GetAdditionalKwargs returns additional provider-specific data.
	GetAdditionalKwargs() map[string]any
}

// BaseMessage contains fields shared by all message types.
type BaseMessage struct {
	Content          string         `json:"content"`
	Name             string         `json:"name,omitempty"`
	ID               string         `json:"id,omitempty"`
	AdditionalKwargs map[string]any `json:"additional_kwargs,omitempty"`
	ResponseMetadata map[string]any `json:"response_metadata,omitempty"`
}

// GetContent returns the text content.
func (m *BaseMessage) GetContent() string { return m.Content }

// GetName returns the name.
func (m *BaseMessage) GetName() string { return m.Name }

// GetAdditionalKwargs returns additional kwargs.
func (m *BaseMessage) GetAdditionalKwargs() map[string]any { return m.AdditionalKwargs }

// HumanMessage represents a message from the user.
type HumanMessage struct {
	BaseMessage
}

// GetType returns MessageTypeHuman.
func (m *HumanMessage) GetType() MessageType { return MessageTypeHuman }

// NewHumanMessage creates a new HumanMessage with the given content.
func NewHumanMessage(content string) *HumanMessage {
	return &HumanMessage{BaseMessage: BaseMessage{Content: content}}
}

// AIMessage represents a message from the AI assistant.
type AIMessage struct {
	BaseMessage
	ToolCalls      []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallChunks []ToolCallChunk `json:"tool_call_chunks,omitempty"`
	UsageMetadata  *UsageMetadata  `json:"usage_metadata,omitempty"`
}

// GetType returns MessageTypeAI.
func (m *AIMessage) GetType() MessageType { return MessageTypeAI }

// NewAIMessage creates a new AIMessage with the given content.
func NewAIMessage(content string) *AIMessage {
	return &AIMessage{BaseMessage: BaseMessage{Content: content}}
}

// NewAIMessageWithToolCalls creates an AIMessage that includes tool calls.
func NewAIMessageWithToolCalls(content string, toolCalls []ToolCall) *AIMessage {
	return &AIMessage{
		BaseMessage: BaseMessage{Content: content},
		ToolCalls:   toolCalls,
	}
}

// UsageMetadata contains token usage information from the provider.
type UsageMetadata struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// SystemMessage represents a system instruction message.
type SystemMessage struct {
	BaseMessage
}

// GetType returns MessageTypeSystem.
func (m *SystemMessage) GetType() MessageType { return MessageTypeSystem }

// NewSystemMessage creates a new SystemMessage with the given content.
func NewSystemMessage(content string) *SystemMessage {
	return &SystemMessage{BaseMessage: BaseMessage{Content: content}}
}

// ToolMessage represents the result of a tool execution.
type ToolMessage struct {
	BaseMessage
	ToolCallID string `json:"tool_call_id"`
}

// GetType returns MessageTypeTool.
func (m *ToolMessage) GetType() MessageType { return MessageTypeTool }

// NewToolMessage creates a new ToolMessage with the given content and tool call ID.
func NewToolMessage(content, toolCallID string) *ToolMessage {
	return &ToolMessage{
		BaseMessage: BaseMessage{Content: content},
		ToolCallID:  toolCallID,
	}
}

// FunctionMessage represents the result of a function call (legacy).
type FunctionMessage struct {
	BaseMessage
}

// GetType returns MessageTypeFunction.
func (m *FunctionMessage) GetType() MessageType { return MessageTypeFunction }

// NewFunctionMessage creates a new FunctionMessage.
func NewFunctionMessage(name, content string) *FunctionMessage {
	return &FunctionMessage{BaseMessage: BaseMessage{Content: content, Name: name}}
}

// GenericMessage represents a generic chat message with a custom role.
type GenericMessage struct {
	BaseMessage
	Role string `json:"role"`
}

// GetType returns MessageTypeGeneric.
func (m *GenericMessage) GetType() MessageType { return MessageTypeGeneric }

// NewGenericMessage creates a new GenericMessage with a custom role.
func NewGenericMessage(role, content string) *GenericMessage {
	return &GenericMessage{
		BaseMessage: BaseMessage{Content: content},
		Role:        role,
	}
}

// Messages is a convenience type for a slice of Message.
type Messages []Message

// GetBufferString formats messages into a string representation.
func GetBufferString(messages []Message, humanPrefix, aiPrefix string) string {
	if humanPrefix == "" {
		humanPrefix = "Human"
	}
	if aiPrefix == "" {
		aiPrefix = "AI"
	}
	var result string
	for i, msg := range messages {
		if i > 0 {
			result += "\n"
		}
		switch msg.GetType() {
		case MessageTypeHuman:
			result += humanPrefix + ": " + msg.GetContent()
		case MessageTypeAI:
			result += aiPrefix + ": " + msg.GetContent()
		case MessageTypeSystem:
			result += "System: " + msg.GetContent()
		case MessageTypeTool:
			result += "Tool: " + msg.GetContent()
		case MessageTypeFunction:
			result += "Function: " + msg.GetContent()
		default:
			result += msg.GetContent()
		}
	}
	return result
}
