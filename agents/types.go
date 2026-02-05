// Package agents provides agent implementations that use language models
// to determine which actions to take.
package agents

import (
	"github.com/LucaLanziani/langchain-go/core"
)

// AgentAction represents a request from the agent to execute a tool.
type AgentAction struct {
	// Tool is the name of the tool to execute.
	Tool string `json:"tool"`

	// ToolInput is the input to pass to the tool (may be JSON).
	ToolInput string `json:"tool_input"`

	// Log is additional information about why this action was taken.
	Log string `json:"log"`

	// MessageLog contains the messages that led to this action.
	MessageLog []core.Message `json:"-"`
}

// AgentFinish represents the final output of an agent.
type AgentFinish struct {
	// ReturnValues contains the final output of the agent.
	ReturnValues map[string]any `json:"return_values"`

	// Log is additional information about the final output.
	Log string `json:"log"`

	// MessageLog contains the messages that led to this finish.
	MessageLog []core.Message `json:"-"`
}

// AgentStep represents one iteration of the agent loop:
// an action that was taken and the observation (tool result) that followed.
type AgentStep struct {
	// Action is the action that was executed.
	Action AgentAction `json:"action"`

	// Observation is the result of executing the action.
	Observation string `json:"observation"`
}

// AgentOutput is the union type returned by an agent's planning step.
// Either Actions or Finish will be populated, not both.
type AgentOutput struct {
	// Actions contains tool calls to execute (may be multiple for parallel tool calling).
	Actions []AgentAction

	// Finish contains the final output if the agent is done.
	Finish *AgentFinish
}
