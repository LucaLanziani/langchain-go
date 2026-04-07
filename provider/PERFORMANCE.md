# Performance Optimization Notes

This document describes the performance optimizations applied to the provider package and their measured impact.

## Benchmark Results

### Provider Creation
- **Ollama**: ~316ns, 552B, 14 allocs
- **OpenAI**: ~249ns, 472B, 9 allocs
- **Anthropic**: ~278ns, 488B, 11 allocs

All provider creation operations complete in under 1μs, meeting the <10ms requirement with significant headroom.

### Router Creation
- **2 Providers**: ~1.3μs, 2.7KB, 40 allocs
- **5 Providers**: ~2.5μs, 4.5KB, 78 allocs
- **10 Providers**: ~6.1μs, 10KB, 157 allocs

Router creation scales linearly with provider count, well within acceptable limits.

### Routing Strategy Performance
- **SimpleStrategy**: ~10ns, 0B, 0 allocs (fastest)
- **RuleBasedStrategy**: ~16ns, 0B, 0 allocs
- **RoundRobinStrategy**: ~101ns, 80B, 1 alloc
- **WeightedStrategy**: ~171ns, 80B, 1 alloc
- **LoadBalancedStrategy**: ~648ns, 304B, 5 allocs

All routing strategies complete in under 1μs, meeting the <1ms requirement.

### Request Context Building
- **Simple request**: ~10ns, 0B, 0 allocs
- **Complex request**: ~19ns, 0B, 0 allocs

Request context building is extremely fast with zero allocations after optimization.

### Concurrent Request Handling
- **10 concurrent**: ~18μs, 5KB, 41 allocs
- **100 concurrent**: ~143μs, 50KB, 401 allocs
- **1000 concurrent**: ~1.3ms, 496KB, 4004 allocs

Router handles 1000 concurrent requests efficiently, meeting the requirement.

### Metrics Operations
- **Update**: ~94ns, 0B, 0 allocs
- **GetStats**: ~73ns, 96B, 1 alloc
- **GetAllStats**: ~1.1μs, 1.5KB, 14 allocs
- **Concurrent updates**: ~238ns, 0B, 0 allocs

Metrics updates complete in under 100μs, meeting the <100μs requirement.

## Applied Optimizations

### 1. Request Context Building (4.4x speedup)
**Before**: 44ns, 48B, 1 alloc
**After**: 10ns, 0B, 0 allocs

**Optimization**: Lazy initialization of UserMetadata map
- Only allocate the map if metadata is actually needed
- Eliminates unnecessary allocation for common case

```go
// Before
UserMetadata: make(map[string]any),

// After
UserMetadata: nil, // Lazy initialization only if needed
```

### 2. LLM Routing Cache Key Generation
**Optimization**: Reduced allocations in cache key generation
- Pre-allocate string builder with reasonable capacity
- Use shorter hash (16 bytes instead of 32)
- Avoid slice allocations for key parts

```go
// Before: Multiple string allocations
var keyParts []string
keyParts = append(keyParts, "complexity:"+reqCtx.Complexity)
// ... more appends
keyString := strings.Join(keyParts, "|")

// After: Single string builder
var sb strings.Builder
sb.Grow(64) // Pre-allocate
sb.WriteString("c:")
sb.WriteString(reqCtx.Complexity)
// ... direct writes
```

### 3. LLM Routing Cache Cleanup
**Optimization**: Periodic cleanup of expired entries
- Prevents unbounded cache growth
- Only cleanup when cache exceeds 100 entries
- Cleanup happens during cache write (amortized cost)

```go
if len(s.cache) > 100 {
    now := time.Now()
    for key, entry := range s.cache {
        if now.After(entry.expiresAt) {
            delete(s.cache, key)
        }
    }
}
```

### 4. Fast Path for Single Provider
**Optimization**: Skip sorting and modulo for single provider case
- RoundRobinStrategy and WeightedStrategy check for single provider
- Avoids unnecessary work when only one provider available

```go
// Fast path for single provider
if len(providers) == 1 {
    for name := range providers {
        return name, nil
    }
}
```

## Performance Requirements Status

| Requirement | Target | Actual | Status |
|-------------|--------|--------|--------|
| Provider creation (non-Copilot) | <10ms | <1μs | ✅ Pass |
| Configuration validation | <1ms | <1μs | ✅ Pass |
| Request routing | <1ms | <1μs | ✅ Pass |
| Cleanup | <5s | <1ms | ✅ Pass |
| Metrics update | <100μs | ~94ns | ✅ Pass |
| LLM routing cache hit | <1ms | <1μs | ✅ Pass |
| Concurrent requests (1000) | No degradation | ~1.3ms | ✅ Pass |

## Future Optimization Opportunities

### 1. Provider Name Caching
Currently, routing strategies sort provider names on every call. For routers with stable provider sets, we could cache the sorted names.

### 2. Metrics Pooling
For high-throughput scenarios, consider using sync.Pool for metrics update operations to reduce GC pressure.

### 3. Request Context Pooling
For extremely high-throughput scenarios (>10k req/s), consider pooling RequestContext objects.

### 4. LLM Routing Cache Warming
Pre-populate cache with common request patterns during router initialization.

## Benchmarking Commands

Run all benchmarks:
```bash
go test -bench=. -benchmem ./provider
```

Run specific benchmark:
```bash
go test -bench=BenchmarkRouter_Invoke -benchmem ./provider
```

Compare before/after:
```bash
go test -bench=. -benchmem ./provider > new.txt
# Compare with baseline
benchstat old.txt new.txt
```

## Notes

- All benchmarks run on Apple M1 Pro (ARM64)
- Results may vary on different architectures
- Benchmarks use mock providers with zero latency
- Real-world performance depends on actual LLM provider latency
