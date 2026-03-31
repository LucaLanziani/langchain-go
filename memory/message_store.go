package memory

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
)

// storedMessage is the JSON-serializable representation of a chat message.
type storedMessage struct {
	Type             string           `json:"type"`
	Content          string           `json:"content"`
	Name             string           `json:"name,omitempty"`
	ToolCalls        []core.ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	AdditionalKwargs map[string]any   `json:"additional_kwargs,omitempty"`
	Timestamp        time.Time        `json:"timestamp"`
}

// messageToStored converts a core.Message to its storedMessage representation.
func messageToStored(msg core.Message) storedMessage {
	sm := storedMessage{
		Type:             string(msg.GetType()),
		Content:          msg.GetContent(),
		Name:             msg.GetName(),
		AdditionalKwargs: msg.GetAdditionalKwargs(),
		Timestamp:        time.Now().UTC(),
	}
	if ai, ok := msg.(*core.AIMessage); ok {
		sm.ToolCalls = ai.ToolCalls
	}
	if tm, ok := msg.(*core.ToolMessage); ok {
		sm.ToolCallID = tm.ToolCallID
	}
	return sm
}

// storedToMessage converts a storedMessage back to a core.Message.
func storedToMessage(sm storedMessage) (core.Message, error) {
	switch core.MessageType(sm.Type) {
	case core.MessageTypeHuman:
		m := core.NewHumanMessage(sm.Content)
		m.Name = sm.Name
		m.AdditionalKwargs = sm.AdditionalKwargs
		return m, nil
	case core.MessageTypeAI:
		m := core.NewAIMessage(sm.Content)
		m.Name = sm.Name
		m.AdditionalKwargs = sm.AdditionalKwargs
		m.ToolCalls = sm.ToolCalls
		return m, nil
	case core.MessageTypeSystem:
		m := core.NewSystemMessage(sm.Content)
		m.Name = sm.Name
		m.AdditionalKwargs = sm.AdditionalKwargs
		return m, nil
	case core.MessageTypeTool:
		m := core.NewToolMessage(sm.Content, sm.ToolCallID)
		m.Name = sm.Name
		m.AdditionalKwargs = sm.AdditionalKwargs
		return m, nil
	case core.MessageTypeFunction:
		m := core.NewFunctionMessage(sm.Name, sm.Content)
		m.AdditionalKwargs = sm.AdditionalKwargs
		return m, nil
	case core.MessageTypeGeneric:
		m := core.NewGenericMessage(sm.Name, sm.Content)
		m.AdditionalKwargs = sm.AdditionalKwargs
		return m, nil
	default:
		return nil, fmt.Errorf("unknown message type: %q", sm.Type)
	}
}

// marshalMessages serializes a slice of messages to JSON.
func marshalMessages(msgs []core.Message) ([]byte, error) {
	stored := make([]storedMessage, len(msgs))
	for i, m := range msgs {
		stored[i] = messageToStored(m)
	}
	return json.Marshal(stored)
}

// unmarshalMessages deserializes JSON into a slice of core.Message.
func unmarshalMessages(data []byte) ([]core.Message, error) {
	var stored []storedMessage
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, err
	}
	msgs := make([]core.Message, 0, len(stored))
	for _, sm := range stored {
		m, err := storedToMessage(sm)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, m)
	}
	return msgs, nil
}
