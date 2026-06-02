package codex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// authClientID is the public OAuth client_id used by the official Codex CLI.
	authClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	// refreshWindow matches the Codex CLI: refresh when the access token is within
	// this duration of expiring.
	refreshWindow = 5 * time.Minute
)

// authTokenURL is OpenAI's OAuth token endpoint used to refresh access tokens.
// It is a var rather than a const so tests can redirect it to httptest.
var authTokenURL = "https://auth.openai.com/oauth/token"

// AuthFile mirrors the on-disk shape of ~/.codex/auth.json.
type AuthFile struct {
	OpenAIAPIKey string     `json:"OPENAI_API_KEY,omitempty"`
	Tokens       *AuthToken `json:"tokens,omitempty"`
	LastRefresh  string     `json:"last_refresh,omitempty"`
}

// AuthToken holds the OAuth tokens issued by ChatGPT login.
type AuthToken struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id,omitempty"`
}

// idTokenClaims is the subset of JWT claims Codex inspects from the id_token.
type idTokenClaims struct {
	Exp  int64               `json:"exp"`
	Auth idTokenAuthSubclaim `json:"https://api.openai.com/auth"`
}

type idTokenAuthSubclaim struct {
	ChatGPTAccountID string `json:"chatgpt_account_id"`
}

// AuthManager loads, caches, and refreshes Codex CLI credentials.
type AuthManager struct {
	path       string
	httpClient *http.Client

	mu    sync.Mutex
	cache *AuthFile
}

// NewAuthManager constructs a manager backed by the file at path. An empty path
// resolves to $CODEX_HOME/auth.json (defaulting to ~/.codex/auth.json).
func NewAuthManager(path string) (*AuthManager, error) {
	if path == "" {
		resolved, err := defaultAuthPath()
		if err != nil {
			return nil, err
		}
		path = resolved
	}
	return &AuthManager{
		path:       path,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// Path returns the auth file path the manager is bound to.
func (m *AuthManager) Path() string { return m.path }

// AccessToken returns a valid bearer token, refreshing if it is close to expiry.
// It also returns the account_id used for the ChatGPT-Account-ID header.
func (m *AuthManager) AccessToken(ctx context.Context) (token, accountID string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cache == nil {
		loaded, err := loadAuthFile(m.path)
		if err != nil {
			return "", "", err
		}
		m.cache = loaded
	}

	if m.cache.Tokens == nil || m.cache.Tokens.AccessToken == "" {
		return "", "", errors.New("codex: no ChatGPT tokens in auth file; run `codex login` first")
	}

	if needsRefresh(m.cache.Tokens.AccessToken) {
		if err := m.refreshLocked(ctx); err != nil {
			return "", "", err
		}
	}

	return m.cache.Tokens.AccessToken, m.cache.Tokens.AccountID, nil
}

func (m *AuthManager) refreshLocked(ctx context.Context) error {
	if m.cache == nil || m.cache.Tokens == nil || m.cache.Tokens.RefreshToken == "" {
		return errors.New("codex: cannot refresh — no refresh_token present")
	}

	body, err := json.Marshal(map[string]string{
		"client_id":     authClientID,
		"grant_type":    "refresh_token",
		"refresh_token": m.cache.Tokens.RefreshToken,
	})
	if err != nil {
		return fmt.Errorf("codex: marshal refresh body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, authTokenURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("codex: build refresh request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("codex: token refresh: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("codex: read refresh response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("codex: token refresh failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var refreshed struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.Unmarshal(respBody, &refreshed); err != nil {
		return fmt.Errorf("codex: parse refresh response: %w", err)
	}
	if refreshed.AccessToken == "" {
		return errors.New("codex: refresh response missing access_token")
	}

	m.cache.Tokens.AccessToken = refreshed.AccessToken
	if refreshed.IDToken != "" {
		m.cache.Tokens.IDToken = refreshed.IDToken
	}
	if refreshed.RefreshToken != "" {
		m.cache.Tokens.RefreshToken = refreshed.RefreshToken
	}
	if accountID, ok := extractAccountID(m.cache.Tokens.IDToken); ok && accountID != "" {
		m.cache.Tokens.AccountID = accountID
	}
	m.cache.LastRefresh = time.Now().UTC().Format(time.RFC3339)

	return saveAuthFile(m.path, m.cache)
}

func defaultAuthPath() (string, error) {
	if home := os.Getenv("CODEX_HOME"); home != "" {
		return filepath.Join(home, "auth.json"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("codex: resolve home dir: %w", err)
	}
	return filepath.Join(home, ".codex", "auth.json"), nil
}

func loadAuthFile(path string) (*AuthFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("codex: auth file %s not found; run `codex login` first", path)
		}
		return nil, fmt.Errorf("codex: read auth file: %w", err)
	}
	var af AuthFile
	if err := json.Unmarshal(data, &af); err != nil {
		return nil, fmt.Errorf("codex: parse auth file: %w", err)
	}
	if af.Tokens != nil && af.Tokens.AccountID == "" {
		if accountID, ok := extractAccountID(af.Tokens.IDToken); ok {
			af.Tokens.AccountID = accountID
		}
	}
	return &af, nil
}

func saveAuthFile(path string, af *AuthFile) error {
	data, err := json.MarshalIndent(af, "", "  ")
	if err != nil {
		return fmt.Errorf("codex: marshal auth file: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("codex: write auth file: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("codex: replace auth file: %w", err)
	}
	return nil
}

func needsRefresh(accessToken string) bool {
	claims, ok := parseJWTClaims(accessToken)
	if !ok || claims.Exp == 0 {
		// If we can't read the expiry, force a refresh so callers can recover
		// from a malformed token instead of hammering an expired one.
		return true
	}
	expiry := time.Unix(claims.Exp, 0)
	return time.Until(expiry) <= refreshWindow
}

func extractAccountID(idToken string) (string, bool) {
	claims, ok := parseJWTClaims(idToken)
	if !ok {
		return "", false
	}
	return claims.Auth.ChatGPTAccountID, claims.Auth.ChatGPTAccountID != ""
}

func parseJWTClaims(token string) (idTokenClaims, bool) {
	parts := strings.Split(token, ".")
	if len(parts) < 2 {
		return idTokenClaims{}, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		// Some issuers emit padded base64 — try the standard decoder as a fallback.
		payload, err = base64.URLEncoding.DecodeString(parts[1])
		if err != nil {
			return idTokenClaims{}, false
		}
	}
	var claims idTokenClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return idTokenClaims{}, false
	}
	return claims, true
}
