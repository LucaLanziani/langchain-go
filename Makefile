.PHONY: help build test test-lmstudio-integration fmt vet lint clean run

LMSTUDIO_HOST ?= 127.0.0.1
LMSTUDIO_PORT ?= 1234
LMSTUDIO_MODEL ?= openai/gpt-oss-20b
LMSTUDIO_AUTH_TOKEN ?= lmstudio

export LMSTUDIO_HOST LMSTUDIO_PORT LMSTUDIO_MODEL LMSTUDIO_AUTH_TOKEN
export LMSTUDIO_OPENAI_BASE_URL LMSTUDIO_OPENAI_MODEL LMSTUDIO_OPENAI_AUTH_TOKEN
export LMSTUDIO_ANTHROPIC_BASE_URL LMSTUDIO_ANTHROPIC_MODEL LMSTUDIO_ANTHROPIC_AUTH_TOKEN

# Default target
help:
	@echo "Available targets:"
	@echo "  make build    - Build the project"
	@echo "  make test     - Run tests"
	@echo "  make test-lmstudio-integration - Run LM Studio provider integration tests"
	@echo "    overrides: LMSTUDIO_HOST, LMSTUDIO_PORT, LMSTUDIO_MODEL, LMSTUDIO_AUTH_TOKEN"
	@echo "  make fmt      - Format code with gofmt"
	@echo "  make vet      - Run go vet"
	@echo "  make lint     - Run golangci-lint (if installed)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Run the main application"

# Build the project
build:
	go build -v ./...

# Run tests
test:
	go test -v -race -coverprofile=coverage.out ./...

# Run LM Studio provider integration tests
test-lmstudio-integration:
	go test -tags=integration ./providers/anthropic ./providers/openai -run TestLMStudio -v

# Format code
fmt:
	go fmt ./...

# Run go vet
vet:
	go vet ./...

# Run linter (requires golangci-lint)
lint:
	@which golangci-lint > /dev/null || (echo "golangci-lint not installed. Install from https://golangci-lint.run/usage/install/" && exit 1)
	golangci-lint run ./...

# Clean build artifacts
clean:
	go clean
	rm -f coverage.out

# Run the main application (adjust path as needed)
run:
	go run ./...
