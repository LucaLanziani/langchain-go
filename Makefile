.PHONY: help build test fmt vet lint clean run release

# Default target
help:
	@echo "Available targets:"
	@echo "  make build    - Build the project"
	@echo "  make test     - Run tests"
	@echo "  make fmt      - Format code with gofmt"
	@echo "  make vet      - Run go vet"
	@echo "  make lint     - Run golangci-lint (if installed)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make run      - Run the main application"
	@echo "  make release  - Create and push a release tag (VERSION=x.y.z)"

# Build the project
build:
	go build -v ./...

# Run tests
test:
	go test -v -race ./...

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

releases:
	git tag --list

# Create and push a release tag
# Usage: make release VERSION=v1.0.0
release: fmt vet test
ifndef VERSION
	@echo "Error: VERSION not specified. Usage: make release VERSION=vX.Y.Z"
	@exit 1
endif
	@echo "Creating release tag: $(VERSION)"
	git tag -a $(VERSION) -m "Release $(VERSION)"
	git push origin $(VERSION)
	@echo "Release $(VERSION) created and pushed successfully"
