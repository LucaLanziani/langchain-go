# Feature Proposals

This folder contains feature proposals for langchain-go. Each feature is described with a user story, acceptance criteria, example usage, and a detailed implementation plan.

## Index

| # | Issue | Feature | Priority | Complexity | Key Benefit |
|---|-------|---------|----------|------------|-------------|
| [001](001-retry-and-rate-limiting.md) | [#1](https://github.com/LucaLanziani/langchain-go/issues/1) | Retry & Rate Limiting Middleware | High | Low-Medium | Production resilience |
| [002](002-document-loaders.md) | [#2](https://github.com/LucaLanziani/langchain-go/issues/2) | Document Loaders | High | Medium | RAG data ingestion |
| [003](003-llm-response-caching.md) | [#3](https://github.com/LucaLanziani/langchain-go/issues/3) | LLM Response Caching | Medium | Low-Medium | Cost reduction & dev speed |
| [005](005-pgvector-store.md) | [#5](https://github.com/LucaLanziani/langchain-go/issues/5) | PostgreSQL pgvector Store | Critical | Medium | Production-grade RAG |
| [006](006-conversation-history-persistence.md) | [#6](https://github.com/LucaLanziani/langchain-go/issues/6) | Persistent Conversation History | Medium | Medium | Multi-turn chatbot production |
| [007](007-structured-output-validation.md) | [#7](https://github.com/LucaLanziani/langchain-go/issues/7) | Structured Output with Validation | Medium | Medium | Type-safe LLM outputs |
| [008](008-observability-and-tracing.md) | [#8](https://github.com/LucaLanziani/langchain-go/issues/8) | Observability (OTel + LangSmith) | Medium | Medium-High | Production monitoring |
| [009](009-map-reduce-summarization.md) | [#9](https://github.com/LucaLanziani/langchain-go/issues/9) | Map-Reduce Summarization | Low-Medium | Medium | Long document processing |
| [010](010-web-search-tools.md) | [#10](https://github.com/LucaLanziani/langchain-go/issues/10) | Web Search & HTTP Tools | Medium | Low-Medium | Real-time agent capabilities |
| [011](011-multimodal-vision-support.md) | [#11](https://github.com/LucaLanziani/langchain-go/issues/11) | Multimodal / Vision Support | Medium | Medium | Image understanding |

## Suggested Implementation Order

1. **001 - Retry & Rate Limiting** — Foundation for all production usage
2. **005 - pgvector Store** — Unlocks production RAG pipelines
3. **002 - Document Loaders** — Completes the RAG data ingestion pipeline
4. **007 - Structured Output** — Improves developer experience for all use cases
5. **011 - Multimodal Vision** — Extends core message model
6. **010 - Web Search Tools** — Gives agents real-time capabilities
7. **003 - LLM Caching** — Cost optimization
8. **006 - Persistent History** — Multi-turn chatbot production-readiness
9. **008 - Observability** — Production monitoring (can be done earlier if needed)
10. **009 - Map-Reduce Summarization** — Specialized chain pattern
