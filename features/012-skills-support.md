## Plan: Add Cross-Provider Skills

Approved for implementation on 2026-04-26. The next action is handoff to an implementation agent; no scope changes requested.

The library does not currently support first-class skills propagation across providers. The recommended approach is to add a model-level `BindSkills` API parallel to `BindTools`, define a provider-neutral `SkillDefinition`, propagate it through `llms.ChatModel` and `provider.Router`, and add provider-local translation hooks in each provider implementation. Per your decisions, the first iteration should cover Anthropic, OpenAI, GitHub Copilot, and Router, and unsupported providers should ignore bound skills silently rather than erroring or falling back to prompt injection.

**Steps**

1. Phase 1: Add the core skill abstraction in the LLM interface layer.
2. Update `/home/lucalanziani/code/langchain-go/llms/chatmodel.go` to add a structured `SkillDefinition` type and `BindSkills(skills ...SkillDefinition) ChatModel` on `llms.ChatModel`.
3. Keep the initial API model-scoped only; do not add provider-factory options in this iteration because the chosen call pattern is binding skills on an already-created model.
4. Phase 2: Propagate the interface through concrete providers and router.
5. Add `boundSkills []llms.SkillDefinition` to the chat models in Anthropic, OpenAI, and GitHub Copilot, next to the existing `boundTools` field.
6. Implement `BindSkills` in each provider using the same copy-on-bind pattern as `BindTools`, so derived models do not alias shared state.
7. Update `/home/lucalanziani/code/langchain-go/provider/router.go` to add `Router.BindSkills` and fan out bound skills to all managed providers, matching the existing `BindTools` and `WithStructuredOutput` behavior.
8. Phase 3: Add provider request/session translation hooks.
9. In Anthropic and OpenAI `buildRequest`, and GitHub Copilot `buildSessionConfig`, add explicit skill handling hooks that read `boundSkills`.
10. Implement provider-local helpers such as `skillsToRequest` or `applySkills` even if the first version is a silent no-op for providers without a concrete request mapping. This creates the extension point where provider-native serialization can be added later without changing the public API.
11. Document in code comments and docs that binding skills does not guarantee provider-side emission in the first iteration.
12. Phase 4: Update tests for interface ripple and binding semantics.
13. Add provider tests mirroring the existing `BindTools` aliasing tests for `BindSkills` in Anthropic, OpenAI, and GitHub Copilot.
14. Add router coverage verifying that `BindSkills` fans out to all providers.
15. Update every test double and mock implementing `llms.ChatModel` so the repo still compiles, including the test helpers in `agents`, `chains`, and `provider`.
16. Phase 5: Document the API and current limits.
17. Update provider-facing docs and examples to show `model.BindSkills(...)` and `router.BindSkills(...)`.
18. Call out the current limitation explicitly: native provider handling is provider-dependent, and the first pass uses silent no-op behavior where no provider-specific mapping exists.

**Relevant files**

- `/home/lucalanziani/code/langchain-go/llms/chatmodel.go` — add `SkillDefinition` and `BindSkills` to the core chat model abstraction.
- `/home/lucalanziani/code/langchain-go/provider/router.go` — add router-level propagation for bound skills.
- `/home/lucalanziani/code/langchain-go/providers/anthropic/chat.go` — store bound skills and add request-level skill handling hook.
- `/home/lucalanziani/code/langchain-go/providers/openai/chat.go` — store bound skills and add request-level skill handling hook.
- `/home/lucalanziani/code/langchain-go/providers/github-copilot/chat.go` — store bound skills and add session-config skill handling hook.
- `/home/lucalanziani/code/langchain-go/providers/anthropic/chat_test.go` — add `BindSkills` isolation tests.
- `/home/lucalanziani/code/langchain-go/providers/openai/chat_test.go` — add `BindSkills` isolation tests.
- `/home/lucalanziani/code/langchain-go/providers/github-copilot/chat_test.go` — add `BindSkills` and session-config coverage.
- `/home/lucalanziani/code/langchain-go/provider/router_stream_test.go` — update stream test doubles to satisfy the extended interface.
- `/home/lucalanziani/code/langchain-go/provider/strategy_llm_test.go` — update routing test doubles to satisfy the extended interface.
- `/home/lucalanziani/code/langchain-go/provider/benchmark_test.go` — update benchmark test doubles to satisfy the extended interface.
- `/home/lucalanziani/code/langchain-go/agents/executor_test.go` — update agent test doubles to satisfy the extended interface.
- `/home/lucalanziani/code/langchain-go/chains/chains_test.go` — update chain test doubles to satisfy the extended interface.
- `/home/lucalanziani/code/langchain-go/doc/providers.md` — document the new API and provider-dependent behavior.
- `/home/lucalanziani/code/langchain-go/README.md` — optionally add a short skills example if the project root README is intended to surface new top-level capabilities.

**Verification**

1. Run targeted interface and router tests: `go test ./llms ./provider`.
2. Run provider package tests: `go test ./providers/anthropic ./providers/openai ./providers/github-copilot`.
3. Run package tests that contain `llms.ChatModel` mocks affected by the new method: `go test ./agents ./chains`.
4. Run a full repo pass after the targeted tests are green: `go test ./...`.
5. Manually verify the public API by constructing a provider, calling `BindSkills`, and confirming that request/session builders preserve the bound skill list without mutating the original model instance.

**Decisions**

- Public API shape: model-level `BindSkills(...)`, parallel to `BindTools(...)`.
- Skill shape: structured definition, not just a string id and not a raw opaque provider payload.
- Initial surface area: Anthropic, OpenAI, GitHub Copilot, and Router.
- Unsupported provider behavior: silent no-op, not explicit error and not prompt fallback.
- Out of scope for this iteration: provider factory options such as `provider.WithSkills(...)`, agent-level auto-binding, and prompt-based skill emulation.

**Further Considerations**

1. The library still needs a concrete provider-native serialization contract for skills; the first iteration should isolate that behind provider-local helper functions so the public API can stabilize before provider mappings do.
2. If callers later need skills at router construction time rather than after creation, add sugar on top of the same core abstraction instead of introducing a second source of truth.
