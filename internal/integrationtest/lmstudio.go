package integrationtest

import (
	"fmt"
	"os"
	"strings"

	"github.com/LucaLanziani/langchain-go/core"
)

const (
	DefaultHost      = "127.0.0.1"
	DefaultPort      = "1234"
	DefaultAuthToken = "lmstudio"
	DefaultModel     = "openai/gpt-oss-20b"
)

func Env(fallback string, keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return fallback
}

func BaseURL(overrideKey, path string) string {
	if value := Env("", overrideKey); value != "" {
		return value
	}

	host := Env(DefaultHost, "LMSTUDIO_HOST")
	port := Env(DefaultPort, "LMSTUDIO_PORT")

	return fmt.Sprintf("http://%s:%s%s", host, port, path)
}

func AuthToken(providerKey string) string {
	return Env(DefaultAuthToken, providerKey, "LMSTUDIO_AUTH_TOKEN")
}

func Model(providerKey string) string {
	return Env(DefaultModel, providerKey, "LMSTUDIO_MODEL")
}

func HasOutput(message *core.AIMessage) bool {
	if message == nil {
		return false
	}

	return strings.TrimSpace(message.GetContent()) != "" || len(message.ToolCalls) > 0
}

func StreamSummary(chunks []*core.AIMessage) (string, int) {
	var content strings.Builder
	toolCalls := 0

	for _, chunk := range chunks {
		if chunk == nil {
			continue
		}

		content.WriteString(chunk.GetContent())
		toolCalls += len(chunk.ToolCalls)
	}

	return content.String(), toolCalls
}
