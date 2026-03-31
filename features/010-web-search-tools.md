# Feature 010: Built-in Web Search and HTTP Tools

## User Story

**As a** developer building an agent that needs access to real-time information,
**I want** pre-built web search tools (Tavily, SerpAPI, DuckDuckGo) and an HTTP fetch tool,
**so that** my agents can search the web, retrieve current information, and access APIs without me writing custom HTTP integration code.

### Acceptance Criteria

- A `TavilySearchTool` wraps the Tavily Search API with search and extract capabilities.
- A `SerpAPITool` wraps the SerpAPI for Google search results.
- A `DuckDuckGoTool` provides free web search without an API key.
- An `HTTPTool` allows agents to fetch content from arbitrary URLs (with configurable allowlists).
- All tools implement the `tools.Tool` interface and work with any agent type.
- Each tool returns structured results (title, URL, snippet) formatted for LLM consumption.
- API keys are configurable via options or environment variables.
- Search tools support options: max results, search depth, include/exclude domains.

### Example Usage

```go
import "github.com/LucaLanziani/langchain-go/tools/search"

// Tavily search (recommended for agents)
tavily := search.NewTavilySearch(
    search.WithTavilyAPIKey("tvly-..."), // or TAVILY_API_KEY env var
    search.WithMaxResults(5),
    search.WithSearchDepth("advanced"),
)

// DuckDuckGo (free, no API key)
ddg := search.NewDuckDuckGo(
    search.WithMaxResults(5),
    search.WithRegion("us-en"),
)

// HTTP fetcher with security controls
httpTool := search.NewHTTPFetch(
    search.WithAllowedDomains([]string{"*.wikipedia.org", "docs.go.dev"}),
    search.WithMaxContentLength(50000),
    search.WithTimeout(10 * time.Second),
)

// Use with an agent
agent := agents.NewToolCallingAgent(model,
    []tools.Tool{tavily, httpTool},
    prompt,
)
exec := agents.NewAgentExecutor(agent, []tools.Tool{tavily, httpTool})

result, _ := exec.Invoke(ctx, map[string]any{
    "input": "What happened in tech news today?",
})
```

---

## Implementation Plan

### New Package: `tools/search/`

#### Tavily: `tools/search/tavily.go`

1. **`TavilySearchTool`** implements `tools.Tool`:
   - `Name()`: `"tavily_search"`
   - `Description()`: `"Search the web for current information. Returns relevant results with titles, URLs, and snippets."`
   - `ArgsSchema()`: `{"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Max results (1-10)"}}`
   - `Run(ctx, input)`:
     - Parse input (string query or JSON with options).
     - POST to `https://api.tavily.com/search` with `{api_key, query, max_results, search_depth, include_domains, exclude_domains}`.
     - Parse response, format as readable text: `"1. [Title](URL)\nSnippet\n\n2. ..."`.

2. **`TavilyExtractTool`** — separate tool for extracting content from URLs via Tavily's extract API.

#### DuckDuckGo: `tools/search/duckduckgo.go`

1. **`DuckDuckGoTool`** implements `tools.Tool`:
   - Uses DuckDuckGo's HTML search (no API key required).
   - Parses search results from the response.
   - Rate-limited internally to respect DuckDuckGo's terms.

#### HTTP Fetch: `tools/search/http.go`

1. **`HTTPFetchTool`** implements `tools.Tool`:
   - `Name()`: `"http_fetch"`
   - `Description()`: `"Fetch the content of a web page by URL. Returns the text content."`
   - `Run(ctx, input)`:
     - Validate URL against allowlist (domain glob matching).
     - Reject private/internal IPs (SSRF protection: no `127.0.0.0/8`, `10.0.0.0/8`, `192.168.0.0/16`, `169.254.0.0/16`, `::1`).
     - GET with configurable timeout and User-Agent.
     - Strip HTML tags, return plain text truncated to `MaxContentLength`.

2. **Security controls**:
   - `WithAllowedDomains([]string)` — glob patterns for allowed domains (required if no explicit allowlist, refuse all).
   - `WithBlockPrivateIPs(bool)` — default: true (SSRF protection).
   - `WithMaxContentLength(int)` — default: 50000 chars.
   - `WithUserAgent(string)` — default: `"langchain-go/1.0"`.

#### SerpAPI: `tools/search/serpapi.go`

1. **`SerpAPITool`** implements `tools.Tool`:
   - GET to `https://serpapi.com/search.json?q={query}&api_key={key}`.
   - Parse `organic_results` array into formatted text.
   - Options: engine (google, bing, yahoo), location, language.

### Result Formatting

All search tools return results in a consistent format:

```
1. **Title** (url)
   Snippet text here...

2. **Title** (url)
   Snippet text here...
```

This format is LLM-friendly and includes enough context for the agent to decide next steps.

### Testing Strategy

- Unit tests with `httptest.NewServer` mocking each search API's response format.
- Test Tavily response parsing with real API response fixtures.
- Test DuckDuckGo HTML parsing with saved HTML fixtures.
- Test HTTP fetch SSRF protection (reject private IPs, respect allowlist).
- Test URL validation and domain glob matching.
- Test content truncation at max length.
- Integration tests (behind build tags) against real APIs.

### Dependencies

- `golang.org/x/net/html` — for HTML text extraction (already needed by Document Loaders feature).
- No other external dependencies.
