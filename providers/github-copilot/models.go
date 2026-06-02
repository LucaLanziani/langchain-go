package copilot

import "context"

// Model describes a model available through the GitHub Copilot CLI bridge.
type Model struct {
	ID                        string   `json:"id"`
	Name                      string   `json:"name"`
	SupportedReasoningEfforts []string `json:"supportedReasoningEfforts,omitempty"`
	DefaultReasoningEffort    string   `json:"defaultReasoningEffort,omitempty"`
}

// ListModels asks the Copilot CLI for the models available to the authenticated user.
func (m *ChatModel) ListModels(ctx context.Context) ([]Model, error) {
	infos, err := m.client.ListModels(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]Model, 0, len(infos))
	for _, info := range infos {
		out = append(out, Model{
			ID:                        info.ID,
			Name:                      info.Name,
			SupportedReasoningEfforts: info.SupportedReasoningEfforts,
			DefaultReasoningEffort:    info.DefaultReasoningEffort,
		})
	}
	return out, nil
}
