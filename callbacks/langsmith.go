package callbacks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
)

// LangSmithHandler sends tracing data to LangSmith for observability.
type LangSmithHandler struct {
	core.BaseCallbackHandler

	apiKey    string
	endpoint  string
	project   string
	client    *http.Client
	runs      map[string]*langSmithRun
	mu        sync.Mutex
	queueOnce sync.Once
	queueWG   sync.WaitGroup
	queue     chan func()
	closed    bool
}

type langSmithRun struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	RunType     string         `json:"run_type"`
	Inputs      map[string]any `json:"inputs,omitempty"`
	Outputs     map[string]any `json:"outputs,omitempty"`
	Error       string         `json:"error,omitempty"`
	StartTime   time.Time      `json:"start_time"`
	EndTime     *time.Time     `json:"end_time,omitempty"`
	ParentRunID string         `json:"parent_run_id,omitempty"`
	SessionName string         `json:"session_name,omitempty"`
	Extra       map[string]any `json:"extra,omitempty"`
}

// NewLangSmithHandler creates a new LangSmith tracing handler.
// It reads LANGCHAIN_API_KEY and LANGCHAIN_ENDPOINT from environment variables.
func NewLangSmithHandler(project string) *LangSmithHandler {
	apiKey := os.Getenv("LANGCHAIN_API_KEY")
	endpoint := os.Getenv("LANGCHAIN_ENDPOINT")
	if endpoint == "" {
		endpoint = "https://api.smith.langchain.com"
	}
	if project == "" {
		project = os.Getenv("LANGCHAIN_PROJECT")
		if project == "" {
			project = "default"
		}
	}

	return &LangSmithHandler{
		apiKey:   apiKey,
		endpoint: endpoint,
		project:  project,
		client:   &http.Client{Timeout: 10 * time.Second},
		runs:     make(map[string]*langSmithRun),
	}
}

func (h *LangSmithHandler) OnChainStart(_ context.Context, inputs map[string]any, runID string, parentRunID string, extras map[string]any) {
	name := "Chain"
	if n, ok := extras["name"]; ok {
		name = fmt.Sprintf("%v", n)
	}
	h.startRun(runID, parentRunID, name, "chain", inputs)
}

func (h *LangSmithHandler) OnChainEnd(_ context.Context, outputs map[string]any, runID string) {
	h.endRun(runID, outputs, "")
}

func (h *LangSmithHandler) OnChainError(_ context.Context, err error, runID string) {
	h.endRun(runID, nil, err.Error())
}

func (h *LangSmithHandler) OnLLMStart(_ context.Context, prompts []string, runID string, parentRunID string, extras map[string]any) {
	h.startRun(runID, parentRunID, "LLM", "llm", map[string]any{"prompts": prompts})
}

func (h *LangSmithHandler) OnChatModelStart(_ context.Context, _ []core.Message, runID string, parentRunID string, extras map[string]any) {
	name := "ChatModel"
	if n, ok := extras["name"]; ok {
		name = fmt.Sprintf("%v", n)
	}
	h.startRun(runID, parentRunID, name, "llm", map[string]any{"messages": "..."})
}

func (h *LangSmithHandler) OnLLMEnd(_ context.Context, output *core.LLMResult, runID string) {
	var outputs map[string]any
	if output != nil {
		outputs = map[string]any{"generations": output.Generations}
	}
	h.endRun(runID, outputs, "")
}

func (h *LangSmithHandler) OnLLMError(_ context.Context, err error, runID string) {
	h.endRun(runID, nil, err.Error())
}

func (h *LangSmithHandler) OnToolStart(_ context.Context, toolName string, input string, runID string, parentRunID string) {
	h.startRun(runID, parentRunID, toolName, "tool", map[string]any{"input": input})
}

func (h *LangSmithHandler) OnToolEnd(_ context.Context, output string, runID string) {
	h.endRun(runID, map[string]any{"output": output}, "")
}

func (h *LangSmithHandler) OnToolError(_ context.Context, err error, runID string) {
	h.endRun(runID, nil, err.Error())
}

func (h *LangSmithHandler) OnRetrieverStart(_ context.Context, query string, runID string, parentRunID string) {
	h.startRun(runID, parentRunID, "Retriever", "retriever", map[string]any{"query": query})
}

func (h *LangSmithHandler) OnRetrieverEnd(_ context.Context, documents []*core.Document, runID string) {
	h.endRun(runID, map[string]any{"documents": len(documents)}, "")
}

func (h *LangSmithHandler) OnRetrieverError(_ context.Context, err error, runID string) {
	h.endRun(runID, nil, err.Error())
}

func (h *LangSmithHandler) startRun(runID, parentRunID, name, runType string, inputs map[string]any) {
	h.initQueue()

	h.mu.Lock()
	run := &langSmithRun{
		ID:          runID,
		Name:        name,
		RunType:     runType,
		Inputs:      core.CloneMap(inputs),
		StartTime:   time.Now().UTC(),
		ParentRunID: parentRunID,
		SessionName: h.project,
	}
	h.runs[runID] = run
	snapshot := cloneLangSmithRun(run)
	h.mu.Unlock()

	h.enqueue(func() {
		h.postRun(snapshot)
	})
}

func (h *LangSmithHandler) endRun(runID string, outputs map[string]any, errMsg string) {
	h.initQueue()

	h.mu.Lock()
	run, ok := h.runs[runID]
	if !ok {
		h.mu.Unlock()
		return
	}
	now := time.Now().UTC()
	run.EndTime = &now
	run.Outputs = core.CloneMap(outputs)
	run.Error = errMsg
	delete(h.runs, runID)
	snapshot := cloneLangSmithRun(run)
	h.mu.Unlock()

	h.enqueue(func() {
		h.patchRun(snapshot)
	})
}

// Flush waits for all queued HTTP work to complete.
func (h *LangSmithHandler) Flush(ctx context.Context) error {
	h.initQueue()

	done := make(chan struct{})
	go func() {
		defer close(done)
		h.queueWG.Wait()
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close prevents new events from being queued and flushes pending work.
func (h *LangSmithHandler) Close(ctx context.Context) error {
	h.mu.Lock()
	h.closed = true
	h.mu.Unlock()
	return h.Flush(ctx)
}

func (h *LangSmithHandler) initQueue() {
	h.queueOnce.Do(func() {
		h.queue = make(chan func(), 128)
		go func() {
			for job := range h.queue {
				job()
				h.queueWG.Done()
			}
		}()
	})
}

func (h *LangSmithHandler) enqueue(job func()) {
	h.mu.Lock()
	closed := h.closed
	h.mu.Unlock()
	if closed {
		return
	}

	h.queueWG.Add(1)
	h.queue <- job
}

func (h *LangSmithHandler) postRun(run *langSmithRun) {
	if h.apiKey == "" {
		return
	}
	data, err := json.Marshal(run)
	if err != nil {
		log.Printf("langsmith: failed to marshal run %s: %v", run.ID, err)
		return
	}
	req, err := http.NewRequest(http.MethodPost, h.endpoint+"/runs", bytes.NewReader(data))
	if err != nil {
		log.Printf("langsmith: failed to create request for run %s: %v", run.ID, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", h.apiKey)

	resp, err := h.client.Do(req)
	if err != nil {
		log.Printf("langsmith: failed to post run %s: %v", run.ID, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("langsmith: API error posting run %s (status %d): %s", run.ID, resp.StatusCode, body)
	}
}

func (h *LangSmithHandler) patchRun(run *langSmithRun) {
	if h.apiKey == "" {
		return
	}
	patchData := map[string]any{
		"end_time": run.EndTime,
		"outputs":  run.Outputs,
	}
	if run.Error != "" {
		patchData["error"] = run.Error
	}
	data, err := json.Marshal(patchData)
	if err != nil {
		log.Printf("langsmith: failed to marshal patch for run %s: %v", run.ID, err)
		return
	}
	req, err := http.NewRequest(http.MethodPatch, h.endpoint+"/runs/"+run.ID, bytes.NewReader(data))
	if err != nil {
		log.Printf("langsmith: failed to create patch request for run %s: %v", run.ID, err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", h.apiKey)

	resp, err := h.client.Do(req)
	if err != nil {
		log.Printf("langsmith: failed to patch run %s: %v", run.ID, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		log.Printf("langsmith: API error patching run %s (status %d): %s", run.ID, resp.StatusCode, body)
	}
}

func cloneLangSmithRun(run *langSmithRun) *langSmithRun {
	if run == nil {
		return nil
	}
	snapshot := *run
	snapshot.Inputs = core.CloneMap(run.Inputs)
	snapshot.Outputs = core.CloneMap(run.Outputs)
	snapshot.Extra = core.CloneMap(run.Extra)
	if run.EndTime != nil {
		end := *run.EndTime
		snapshot.EndTime = &end
	}
	return &snapshot
}
