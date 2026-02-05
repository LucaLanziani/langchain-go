package agents

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/langchain-go/langchain-go/core"
	"github.com/langchain-go/langchain-go/llms"
	"github.com/langchain-go/langchain-go/prompts"
	"github.com/langchain-go/langchain-go/tools"
)

// ReAct agent output parsing regex.
var (
	actionRegex      = regexp.MustCompile(`Action\s*:\s*(.+?)(?:\n|$)`)
	actionInputRegex = regexp.MustCompile(`Action\s*Input\s*:\s*(.+?)(?:\n|$)`)
	finalAnswerRegex = regexp.MustCompile(`Final\s*Answer\s*:\s*(.+)`)
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
		sb.WriteString(fmt.Sprintf("%s: %s\n", t.Name(), t.Description()))
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
		sb.WriteString(fmt.Sprintf("\nObservation: %s\nThought: ", step.Observation))
	}
	return []core.Message{core.NewAIMessage(sb.String())}
}

// parseReActOutput parses the LLM text output into an AgentOutput.
func parseReActOutput(text string) (*AgentOutput, error) {
	// Check for Final Answer.
	if matches := finalAnswerRegex.FindStringSubmatch(text); len(matches) > 1 {
		return &AgentOutput{
			Finish: &AgentFinish{
				ReturnValues: map[string]any{
					"output": strings.TrimSpace(matches[1]),
				},
				Log: text,
			},
		}, nil
	}

	// Check for Action + Action Input.
	actionMatches := actionRegex.FindStringSubmatch(text)
	inputMatches := actionInputRegex.FindStringSubmatch(text)

	if len(actionMatches) > 1 {
		tool := strings.TrimSpace(actionMatches[1])
		toolInput := ""
		if len(inputMatches) > 1 {
			toolInput = strings.TrimSpace(inputMatches[1])
		}
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

// Ensure ReActAgent implements Agent.
var _ Agent = (*ReActAgent)(nil)
