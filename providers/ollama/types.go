package ollama

import "encoding/json"

// chatRequest is the request body for /api/chat.
type chatRequest struct {
	Model     string          `json:"model"`
	Messages  []ollamaMessage `json:"messages"`
	Stream    bool            `json:"stream"`
	Tools     []ollamaTool    `json:"tools,omitempty"`
	Format    string          `json:"format,omitempty"`
	KeepAlive string          `json:"keep_alive,omitempty"`
	Options   *modelOptions   `json:"options,omitempty"`
}

// modelOptions holds Ollama model parameters.
type modelOptions struct {
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	NumCtx      *int     `json:"num_ctx,omitempty"`
}

// ollamaMessage represents a single message in the Ollama chat format.
type ollamaMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Images     []string         `json:"images,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// ollamaTool represents a tool definition for Ollama's tool-calling format.
type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

// ollamaToolFunction holds the function metadata within a tool definition.
type ollamaToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ollamaToolCall represents a tool invocation returned by the model.
type ollamaToolCall struct {
	Function ollamaToolCallFunction `json:"function"`
}

// ollamaToolCallFunction holds the function name and arguments of a tool call.
type ollamaToolCallFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// chatResponse is the non-streaming response from /api/chat.
type chatResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Message            ollamaMessage `json:"message"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	TotalDuration      int64         `json:"total_duration,omitempty"`
	LoadDuration       int64         `json:"load_duration,omitempty"`
	PromptEvalDuration int64         `json:"prompt_eval_duration,omitempty"`
	EvalDuration       int64         `json:"eval_duration,omitempty"`
}

// streamChunk is a single chunk from an NDJSON streaming response.
type streamChunk struct {
	Model     string        `json:"model"`
	CreatedAt string        `json:"created_at"`
	Message   ollamaMessage `json:"message"`
	Done      bool          `json:"done"`
	// Final-chunk fields (present when done == true)
	PromptEvalCount int `json:"prompt_eval_count,omitempty"`
	EvalCount       int `json:"eval_count,omitempty"`
}

// embedRequest is the request body for /api/embed.
type embedRequest struct {
	Model     string   `json:"model"`
	Input     []string `json:"input"`
	KeepAlive string   `json:"keep_alive,omitempty"`
}

// embedResponse is the response from /api/embed.
type embedResponse struct {
	Model      string      `json:"model"`
	Embeddings [][]float64 `json:"embeddings"`
}
