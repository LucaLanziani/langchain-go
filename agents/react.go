package agents

import (
	"context"
	"fmt"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
	"github.com/LucaLanziani/langchain-go/llms"
	"github.com/LucaLanziani/langchain-go/prompts"
	"github.com/LucaLanziani/langchain-go/tools"
)

// DefaultReActPrompt returns the default ReAct prompt template.
func DefaultReActPrompt() *prompts.ChatPromptTemplate {
	return prompts.NewChatPromptTemplate(
		prompts.System(`Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!`),
		prompts.Placeholder("agent_scratchpad"),
		prompts.Human("{input}"),
	)
}

// ReActAgent uses the ReAct (Reasoning + Acting) prompting pattern.
type ReActAgent struct {
	llm    llms.ChatModel
	prompt *prompts.ChatPromptTemplate
	tools  []tools.Tool
}

// NewReActAgent creates a new ReAct agent.
// If prompt is nil, the default ReAct prompt is used.
func NewReActAgent(llm llms.ChatModel, agentTools []tools.Tool, prompt *prompts.ChatPromptTemplate) *ReActAgent {
	if prompt == nil {
		prompt = DefaultReActPrompt()
	}
	return &ReActAgent{
		llm:    llm,
		prompt: prompt,
		tools:  agentTools,
	}
}

// Plan decides the next action based on intermediate steps and inputs.
func (a *ReActAgent) Plan(ctx context.Context, intermediateSteps []AgentStep, inputs map[string]any) (*AgentOutput, error) {
	// Build tool descriptions and names.
	toolDescs := a.renderToolDescriptions()
	toolNames := a.renderToolNames()

	// Build scratchpad from intermediate steps.
	scratchpad := formatReActScratchpad(intermediateSteps)

	// Merge inputs.
	mergedInputs := make(map[string]any)
	for k, v := range inputs {
		mergedInputs[k] = v
	}
	mergedInputs["tools"] = toolDescs
	mergedInputs["tool_names"] = toolNames
	mergedInputs["agent_scratchpad"] = scratchpad

	// Format prompt.
	messages, err := a.prompt.FormatMessages(mergedInputs)
	if err != nil {
		return nil, fmt.Errorf("failed to format prompt: %w", err)
	}

	// Call the model with stop sequences.
	response, err := a.llm.Invoke(ctx, messages, core.WithStop("\nObservation:"))
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// Parse the output.
	return parseReActOutput(response.Content)
}

// InputKeys returns the expected input keys.
func (a *ReActAgent) InputKeys() []string {
	return []string{"input"}
}

// OutputKeys returns the output keys.
func (a *ReActAgent) OutputKeys() []string {
	return []string{"output"}
}

func (a *ReActAgent) renderToolDescriptions() string {
	var sb strings.Builder
	for _, t := range a.tools {
		sb.WriteString(t.Name())
		sb.WriteString(": ")
		sb.WriteString(t.Description())
		sb.WriteByte('\n')
	}
	return sb.String()
}

func (a *ReActAgent) renderToolNames() string {
	names := make([]string, len(a.tools))
	for i, t := range a.tools {
		names[i] = t.Name()
	}
	return strings.Join(names, ", ")
}

// formatReActScratchpad converts intermediate steps to the ReAct text format.
func formatReActScratchpad(steps []AgentStep) []core.Message {
	if len(steps) == 0 {
		return nil
	}
	var sb strings.Builder
	for _, step := range steps {
		sb.WriteString(step.Action.Log)
		sb.WriteString("\nObservation: ")
		sb.WriteString(step.Observation)
		sb.WriteString("\nThought: ")
	}
	return []core.Message{core.NewAIMessage(sb.String())}
}

// parseReActOutput parses the LLM text output into an AgentOutput.
func parseReActOutput(text string) (*AgentOutput, error) {
	if finalAnswer := extractReActSection(text, "Final Answer:", nil); finalAnswer != "" {
		return &AgentOutput{
			Finish: &AgentFinish{
				ReturnValues: map[string]any{
					"output": finalAnswer,
				},
				Log: text,
			},
		}, nil
	}

	tool := extractReActSection(text, "Action:", []string{"Action Input:", "Observation:", "Thought:", "Final Answer:"})
	if tool != "" {
		toolInput := extractReActSection(text, "Action Input:", []string{"Observation:", "Thought:", "Final Answer:", "Action:"})
		return &AgentOutput{
			Actions: []AgentAction{
				{
					Tool:      tool,
					ToolInput: toolInput,
					Log:       text,
				},
			},
		}, nil
	}

	return nil, fmt.Errorf("could not parse LLM output: %q", text)
}

func extractReActSection(text string, prefix string, stopPrefixes []string) string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	collecting := false
	collected := make([]string, 0, len(lines))

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !collecting {
			if strings.HasPrefix(trimmed, prefix) {
				collecting = true
				collected = append(collected, strings.TrimSpace(strings.TrimPrefix(trimmed, prefix)))
			}
			continue
		}

		if hasReActPrefix(trimmed, stopPrefixes) {
			break
		}
		collected = append(collected, line)
	}

	value := strings.TrimSpace(strings.Join(collected, "\n"))
	if value == "" {
		return ""
	}

	valueLines := strings.Split(value, "\n")
	if len(valueLines) >= 2 && strings.HasPrefix(strings.TrimSpace(valueLines[0]), "```") && strings.TrimSpace(valueLines[len(valueLines)-1]) == "```" {
		value = strings.TrimSpace(strings.Join(valueLines[1:len(valueLines)-1], "\n"))
	}

	return value
}

func hasReActPrefix(line string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(line, prefix) {
			return true
		}
	}
	return false
}

// Ensure ReActAgent implements Agent.
var _ Agent = (*ReActAgent)(nil)
