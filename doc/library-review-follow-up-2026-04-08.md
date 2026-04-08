# Library Review Follow-Up - 2026-04-08

## Scope

This document is the remediation follow-up to `doc/library-review-2026-04-08.md`.
Its goals are:

- close every numbered finding from the original review,
- address the package-by-package notes explicitly, and
- record the validation run after the fixes landed.

## Outcome Summary

- All 17 verified findings from the original review were addressed in code.
- Additional package-note follow-ups were also completed for `memory`, `outputparsers`, and router metrics.
- Validation passed with `go test -count=1 ./...`, `go vet ./...`, and a focused `go test -race -count=1 ./core ./provider ./callbacks ./agents ./runnable` run.

## Findings Closure

| #   | Original finding                                                      | Status | Resolution                                                                                                                                                                                            |
| --- | --------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Provider copy helpers alias mutable state                             | Fixed  | `providers/openai`, `providers/anthropic`, `providers/github-copilot`, and `providers/ollama` now deep-copy bound tools and structured-output schema state instead of sharing mutable slices or maps. |
| 2   | `Batch` semantics do not match the documented contract                | Fixed  | Added shared `core.Batch`, migrated library implementations to it, aligned `Runnable.Batch` docs, and honored `core.WithMaxConcurrency` consistently.                                                 |
| 3   | Tool-calling agent scratchpad reuses tool call IDs                    | Fixed  | `agents.ToolCallingAgent` now preserves provider-issued tool call IDs and generates unique fallback IDs when rebuilding the scratchpad.                                                               |
| 4   | Router stream metrics only measure stream startup                     | Fixed  | Router stream metrics now update at terminal stream completion and distinguish success, error, and cancellation with end-to-end latency.                                                              |
| 5   | `WeightedStrategy` uses a non-thread-safe RNG under a read lock       | Fixed  | Weighted routing now serializes RNG access safely.                                                                                                                                                    |
| 6   | LLM-based routing cache is safe but not request-coalescing            | Fixed  | Added in-flight request coalescing so identical concurrent cache misses share a single routing-model evaluation.                                                                                      |
| 7   | OpenAI streaming drops scanner read errors                            | Fixed  | `providers/openai` now surfaces scanner read failures to the stream consumer.                                                                                                                         |
| 8   | Anthropic extended thinking overrides user sampling settings silently | Fixed  | Anthropic thinking mode now validates incompatible sampling settings and returns an explicit error instead of silently overriding them.                                                               |
| 9   | Anthropic non-streaming responses discard thinking blocks             | Fixed  | Non-streaming Anthropic responses now preserve thinking text in `AdditionalKwargs["thinking"]`.                                                                                                       |
| 10  | `ChatPromptTemplate` does not validate missing variables              | Fixed  | Chat prompt formatting now validates required variables consistently with `PromptTemplate`.                                                                                                           |
| 11  | ReAct parsing is too brittle for real-world outputs                   | Fixed  | Replaced the regex-only parser with line-based section extraction that handles multiline tool input, fenced JSON, and final answers more reliably.                                                    |
| 12  | In-memory vector store generates IDs but does not write them back     | Fixed  | Generated IDs are now assigned back onto the source `Document` before storage so callers can rely on stable IDs after insertion.                                                                      |
| 13  | Router batch returns only the first error and no per-item status      | Fixed  | Router batch now returns partial successes plus `*provider.BatchError` with failed item indexes.                                                                                                      |
| 14  | LangSmith tracing is fire-and-forget with no flush or backpressure    | Fixed  | LangSmith tracing now uses a bounded queue, exposes `Flush` and `Close`, and snapshots queued runs so background POST and PATCH work cannot race on shared state.                                     |
| 15  | `NewParallel` claims stable key order, but map input is random        | Fixed  | `runnable.NewParallel`, `runnable.NewParallelAny`, and `runnable.NewAssign` now sort keys for deterministic ordering.                                                                                 |
| 16  | Structured tool schema generation is shallow                          | Fixed  | Tool schema generation is now recursive for nested structs, slices, arrays, maps, and supports `format` and `enum` tags while preserving the top-level object fallback.                               |
| 17  | Provider test coverage is uneven                                      | Fixed  | Added dedicated tests for `providers/openai`, `providers/anthropic`, router metrics, weighted routing, tool-calling IDs, embeddings, vectorstores, and provider copy semantics.                       |

## Package Note Disposition

### core

- Addressed: `Batch` contract is now implemented through a shared helper instead of being only documented.
- Addressed: `StreamIterator` now exposes `Done()` and `Next()` observes closure correctly, which made stream-lifecycle instrumentation viable.

### runnable

- Addressed: `Sequence`, `Branch`, `Lambda`, `Parallel`, `Assign`, and related components now use the shared batch helper.
- Addressed: branch ordering is deterministic after sorting keys.

### prompts

- Addressed: `ChatPromptTemplate` now fails on missing required variables instead of silently formatting partial prompts.

### outputparsers

- Addressed: `Batch` now uses the shared batch helper.
- Addressed: JSON parser errors now truncate long raw payloads instead of embedding arbitrarily large output in the error string.
- Retained tradeoff: schema-aware semantic validation beyond Go unmarshalling is still future work, not a correctness regression.

### tools

- Addressed: generated JSON Schema is recursive for nested objects and richer tags.
- Preserved behavior: non-struct top-level inputs still fall back to an object-shaped schema for backward compatibility.

### agents

- Addressed: tool-call ID reuse is fixed.
- Addressed: ReAct parsing is hardened and covered with multiline and JSON-shaped tests.
- Retained tradeoff: `handleParsingErrors` still follows the bounded retry-until-`maxIterations` model rather than introducing a new retry policy.

### chains

- Addressed: chain `Batch` implementations now honor the shared concurrent contract.
- Addressed: `RetrievalQA` now mirrors `query` into `input` when needed so stricter prompt validation does not break existing chain prompts.

### callbacks

- Addressed: LangSmith delivery now has bounded queueing, flush/close support, and race-safe snapshots.
- Retained tradeoff: callback panic isolation remains unchanged.

### memory

- Addressed: `ConversationWindowMemory` now treats `K <= 0` as an empty window instead of allowing negative values to panic.

### retrievers and vectorstores

- Addressed: generated vector-store IDs are written back to caller-visible documents.
- Retained tradeoff: the in-memory vector store remains an O(n) and unbounded baseline implementation by design.

### provider and routing layer

- Addressed: stream metrics now include completion semantics and cancellation counts.
- Addressed: weighted routing RNG access is concurrency-safe.
- Addressed: LLM-routing cache misses are coalesced.
- Addressed: batch failures now preserve per-item status.
- Addressed: computed stats and reset paths now include the new cancellation accounting.

### providers/openai

- Addressed: mutable state aliasing is removed.
- Addressed: stream read errors are surfaced.
- Addressed: dedicated package tests now exist.

### providers/anthropic

- Addressed: mutable state aliasing is removed.
- Addressed: thinking configuration is validated explicitly.
- Addressed: non-stream thinking output is preserved.
- Addressed: dedicated package tests now exist.

### providers/github-copilot

- Addressed: mutable state aliasing is removed.
- Addressed: runtime `core.WithMaxConcurrency` now takes precedence for batch execution.
- Partially constrained by environment: coverage is still more limited than fully local providers because live integration behavior remains environment-dependent, but clone-regression tests were added.

### providers/ollama

- Addressed: structured-output schema aliasing is removed.
- Addressed: batch execution now uses the shared helper.
- Retained tradeoff: startup validation remains intentionally request-time rather than eagerly probing the server.

## Validation

The remediation set was validated with:

- `gofmt -w` on all changed Go files
- `go test -count=1 ./...`
- `go vet ./...`
- `go test -race -count=1 ./core ./provider ./callbacks ./agents ./runnable`

All of the above completed successfully.

## Residual Notes

The original review mixed correctness bugs with longer-horizon product tradeoffs. After this remediation wave, the remaining items are product choices rather than open correctness defects from the report:

- output-parser schema-aware validation beyond Go unmarshalling,
- callback panic isolation policy,
- in-memory vector-store scaling characteristics,
- `AgentExecutor` parsing-error retry policy, and
- Ollama startup validation strategy.

Those items were not skipped; they are explicitly retained as future design decisions rather than unresolved regressions.
