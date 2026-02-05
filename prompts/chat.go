package prompts

import (
	"context"
	"fmt"
	"strings"

	"github.com/langchain-go/langchain-go/core"
)

// MessageTemplate represents a single message template within a ChatPromptTemplate.
type MessageTemplate struct {
	// Role is the message role (system, human, ai, placeholder).
	Role string

	// Template is the content template with {variable} placeholders.
	// For placeholders, this is the variable name to pull messages from.
	Template string

	// Name is an optional name for the message.
	Name string
}

// ChatPromptTemplate formats a sequence of messages from templates and variables.
// It implements Runnable[map[string]any, []core.Message].
type ChatPromptTemplate struct {
	// Messages are the message templates in order.
	Messages []MessageTemplate

	// InputVariables is the list of all variable names required.
	InputVariables []string

	// PartialVariables are pre-filled variables.
	PartialVariables map[string]any

	name string
}

// NewChatPromptTemplate creates a ChatPromptTemplate from message templates.
func NewChatPromptTemplate(messages ...MessageTemplate) *ChatPromptTemplate {
	seen := make(map[string]bool)
	var vars []string
	for _, msg := range messages {
		if msg.Role == "placeholder" {
			if !seen[msg.Template] {
				seen[msg.Template] = true
				vars = append(vars, msg.Template)
			}
			continue
		}
		for _, v := range extractVariables(msg.Template) {
			if !seen[v] {
				seen[v] = true
				vars = append(vars, v)
			}
		}
	}
	return &ChatPromptTemplate{
		Messages:       messages,
		InputVariables: vars,
	}
}

// FromMessages is an alias for NewChatPromptTemplate.
func FromMessages(messages ...MessageTemplate) *ChatPromptTemplate {
	return NewChatPromptTemplate(messages...)
}

// WithName sets the name for tracing.
func (c *ChatPromptTemplate) WithName(name string) *ChatPromptTemplate {
	c.name = name
	return c
}

// WithPartialVariables sets partial variables.
func (c *ChatPromptTemplate) WithPartialVariables(vars map[string]any) *ChatPromptTemplate {
	c.PartialVariables = vars
	return c
}

// GetName returns the name of this chat prompt template.
func (c *ChatPromptTemplate) GetName() string {
	if c.name != "" {
		return c.name
	}
	return "ChatPromptTemplate"
}

// System creates a system message template.
func System(template string) MessageTemplate {
	return MessageTemplate{Role: "system", Template: template}
}

// Human creates a human message template.
func Human(template string) MessageTemplate {
	return MessageTemplate{Role: "human", Template: template}
}

// AI creates an AI message template.
func AI(template string) MessageTemplate {
	return MessageTemplate{Role: "ai", Template: template}
}

// Placeholder creates a messages placeholder that will be replaced with
// a list of messages from the input variables.
func Placeholder(variableName string) MessageTemplate {
	return MessageTemplate{Role: "placeholder", Template: variableName}
}

// FormatMessages applies the variables and returns formatted messages.
func (c *ChatPromptTemplate) FormatMessages(values map[string]any) ([]core.Message, error) {
	merged := make(map[string]any)
	for k, v := range c.PartialVariables {
		merged[k] = v
	}
	for k, v := range values {
		merged[k] = v
	}

	var messages []core.Message
	for _, tmpl := range c.Messages {
		switch tmpl.Role {
		case "placeholder":
			// Pull messages from the input variable.
			msgVal, ok := merged[tmpl.Template]
			if !ok {
				// Placeholder variables are optional; skip if not provided.
				continue
			}
			msgs, ok := msgVal.([]core.Message)
			if !ok {
				return nil, fmt.Errorf("placeholder variable %q must be []core.Message, got %T", tmpl.Template, msgVal)
			}
			messages = append(messages, msgs...)

		case "system":
			content, err := formatTemplate(tmpl.Template, merged)
			if err != nil {
				return nil, err
			}
			messages = append(messages, core.NewSystemMessage(content))

		case "human":
			content, err := formatTemplate(tmpl.Template, merged)
			if err != nil {
				return nil, err
			}
			messages = append(messages, core.NewHumanMessage(content))

		case "ai":
			content, err := formatTemplate(tmpl.Template, merged)
			if err != nil {
				return nil, err
			}
			messages = append(messages, core.NewAIMessage(content))

		default:
			content, err := formatTemplate(tmpl.Template, merged)
			if err != nil {
				return nil, err
			}
			messages = append(messages, core.NewGenericMessage(tmpl.Role, content))
		}
	}
	return messages, nil
}

// Invoke formats the template with the given input and returns messages.
func (c *ChatPromptTemplate) Invoke(ctx context.Context, input map[string]any, opts ...core.Option) ([]core.Message, error) {
	return c.FormatMessages(input)
}

// Stream returns a single-chunk stream of the formatted messages.
func (c *ChatPromptTemplate) Stream(ctx context.Context, input map[string]any, opts ...core.Option) (*core.StreamIterator[[]core.Message], error) {
	result, err := c.FormatMessages(input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[[]core.Message], 1)
	ch <- core.StreamChunk[[]core.Message]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch formats the template with multiple input maps.
func (c *ChatPromptTemplate) Batch(ctx context.Context, inputs []map[string]any, opts ...core.Option) ([][]core.Message, error) {
	results := make([][]core.Message, len(inputs))
	for i, input := range inputs {
		result, err := c.FormatMessages(input)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// formatTemplate replaces {variable} placeholders in a template string.
func formatTemplate(template string, values map[string]any) (string, error) {
	result := template
	for k, v := range values {
		// Skip non-string values that might be message lists.
		switch v.(type) {
		case []core.Message:
			continue
		}
		result = strings.ReplaceAll(result, "{"+k+"}", fmt.Sprintf("%v", v))
	}
	return result, nil
}
