# Unified Provider Interface Examples

This directory contains 12 complete example programs demonstrating the unified provider interface and router capabilities.

## Prerequisites

Before running these examples, ensure you have:

1. **API Keys** (set as environment variables):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GITHUB_TOKEN="your-github-token"  # For Copilot examples
   ```

2. **LM Studio** (for local model examples):
   ```bash
    # Start LM Studio's local server on port 1234
    # Then load a model and note its identifier
    # The examples use the OpenAI-compatible endpoint at http://localhost:1234/v1
   ```

## Examples Overview

### Basic Examples

#### 1. Single Provider Creation
**Directory:** `01_single_provider/`

Demonstrates creating individual providers (OpenAI, Anthropic, LM Studio, Copilot) using the unified interface.

```bash
cd 01_single_provider
go run main.go
```

**Key Concepts:**
- Creating providers with `NewProvider()`
- Common configuration options
- Provider-specific options
- Resource cleanup

---

#### 2. Provider Switching
**Directory:** `02_provider_switching/`

Shows how easy it is to switch between providers using the same code.

```bash
cd 02_provider_switching

# Try different providers
LLM_PROVIDER=openai go run main.go
LLM_PROVIDER=anthropic go run main.go
LLM_PROVIDER=lmstudio go run main.go
LLM_PROVIDER=copilot go run main.go
```

**Key Concepts:**
- Provider-agnostic code
- Runtime provider selection
- Consistent interface across providers

---

### Router Examples

#### 3. Simple Router
**Directory:** `03_simple_router/`

Creates a router with multiple providers using SimpleStrategy to always route to a specific provider.

```bash
cd 03_simple_router
go run main.go
```

**Key Concepts:**
- Creating a router with multiple providers
- SimpleStrategy for fixed routing
- Router metrics tracking

---

#### 4. Round-Robin Load Balancing
**Directory:** `04_round_robin/`

Distributes requests evenly across providers in a circular pattern.

```bash
cd 04_round_robin
go run main.go
```

**Key Concepts:**
- RoundRobinStrategy for even distribution
- Load balancing across providers
- Metrics showing distribution

---

#### 5. Weighted Routing
**Directory:** `05_weighted_routing/`

Routes requests according to provider weights (e.g., 70% to fast provider, 20% to smart, 10% to a local LM Studio model).

```bash
cd 05_weighted_routing
go run main.go
```

**Key Concepts:**
- WeightedStrategy for proportional distribution
- Cost optimization (prefer cheaper providers)
- Quality vs. cost trade-offs

---

#### 6. Rule-Based Routing
**Directory:** `06_rule_based_routing/`

Routes requests based on characteristics like complexity, token count, and priority.

```bash
cd 06_rule_based_routing
go run main.go
```

**Key Concepts:**
- RuleBasedStrategy with custom conditions
- Priority-based rule evaluation
- Request context analysis
- Default fallback routing

---

#### 7. Load-Balanced Routing
**Directory:** `07_load_balanced_routing/`

Automatically routes to the provider with the best performance metrics.

```bash
cd 07_load_balanced_routing
go run main.go
```

**Key Concepts:**
- LoadBalancedStrategy for performance optimization
- Latency-based routing
- Error rate consideration
- Adaptive load distribution

---

#### 8. Custom Routing Logic
**Directory:** `08_custom_routing/`

Implements custom routing logic based on request content analysis.

```bash
cd 08_custom_routing
go run main.go
```

**Key Concepts:**
- CustomStrategy with user-defined logic
- Content-based routing
- Keyword analysis
- Success/error callbacks

---

#### 9. LLM-Based Intelligent Routing
**Directory:** `09_llm_routing/`

Uses an LLM to analyze requests and intelligently select the most appropriate provider.

```bash
cd 09_llm_routing
go run main.go
```

**Key Concepts:**
- LLMRoutingStrategy for intelligent decisions
- Provider descriptions and capabilities
- Routing decision caching
- Graceful fallback on LLM failure

---

#### 10. Router with Fallback
**Directory:** `10_router_fallback/`

Demonstrates automatic fallback to alternative providers when the primary fails.

```bash
cd 10_router_fallback
go run main.go
```

**Key Concepts:**
- SequentialFallback strategy
- High availability through redundancy
- Automatic retry with alternatives
- SmartFallback based on metrics

---

#### 11. Multiple Instances of Same Provider
**Directory:** `11_multiple_instances/`

Creates multiple instances of the same provider with different configurations.

```bash
cd 11_multiple_instances
go run main.go
```

**Key Concepts:**
- Multiple configurations per provider
- Temperature-based routing (fast/smart/creative)
- Cost and quality optimization
- Independent metrics per instance

---

#### 12. Concurrent Requests
**Directory:** `12_concurrent_requests/`

Demonstrates thread-safe concurrent request handling with automatic load distribution.

```bash
cd 12_concurrent_requests
go run main.go
```

**Key Concepts:**
- Thread-safe router usage
- High-throughput concurrent requests
- Atomic metrics updates
- Performance under load

---

## Running All Examples

To run all examples in sequence:

```bash
#!/bin/bash
for dir in */; do
    if [ -f "$dir/main.go" ]; then
        echo "Running $dir..."
        (cd "$dir" && go run main.go)
        echo ""
    fi
done
```

## Common Patterns

### Creating a Single Provider

```go
model, cleanup, err := provider.NewProvider(
    ctx,
    provider.ProviderOpenAI,
    provider.WithModel("gpt-4o-mini"),
    provider.WithTemperature(0.7),
)
if err != nil {
    log.Fatal(err)
}
defer cleanup()
```

### Creating a Router

```go
router, err := provider.NewRouter(
    ctx,
    []provider.ProviderEntry{
        {
            Name:         "fast",
            ProviderType: provider.ProviderOpenAI,
            Options:      []provider.ProviderOption{...},
        },
        {
            Name:         "smart",
            ProviderType: provider.ProviderAnthropic,
            Options:      []provider.ProviderOption{...},
        },
    },
    strategy,
    provider.WithFallback(fallbackStrategy),
)
if err != nil {
    log.Fatal(err)
}
defer router.Cleanup()
```

### Using the Router

```go
// Router implements llms.ChatModel interface
response, err := router.Invoke(ctx, messages)
if err != nil {
    log.Fatal(err)
}

fmt.Println(response.GetContent())
```

### Checking Metrics

```go
metrics := router.GetMetrics()
for name, m := range metrics {
    fmt.Printf("%s: %d requests, %d errors\n", 
        name, m.RequestCount, m.ErrorCount)
}
```

## Troubleshooting

### LM Studio Connection Issues

If you see errors connecting to LM Studio:

```bash
# Check if the local server is reachable
curl http://localhost:1234/v1/models

# Start LM Studio's local server and load a model if needed
```

### API Key Issues

If you see authentication errors:

```bash
# Verify environment variables are set
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GITHUB_TOKEN

# Set them if missing
export OPENAI_API_KEY="your-key-here"
```

### GitHub Copilot Issues

For Copilot examples, ensure:

```bash
# GitHub CLI is installed and authenticated
gh auth status

# Get a token
gh auth token
```

## Next Steps

After exploring these examples:

1. **Read the Documentation**: See `provider/doc.go` for complete API documentation
2. **Review the Design**: Check `.kiro/specs/unified-provider-interface/design.md`
3. **Explore Tests**: Look at `provider/*_test.go` for more usage patterns
4. **Build Your Own**: Combine patterns to create custom routing strategies

## Support

For issues or questions:
- Check the main README.md
- Review the design document
- Look at test files for more examples
- Open an issue on GitHub
