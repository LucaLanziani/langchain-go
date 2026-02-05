package prompts

// MessagesPlaceholder creates a placeholder MessageTemplate for injecting
// a list of messages into a ChatPromptTemplate.
//
// Usage:
//
//	prompt := NewChatPromptTemplate(
//	    System("You are a helpful assistant."),
//	    MessagesPlaceholder("chat_history"),
//	    Human("{input}"),
//	)
//
// When invoked, the "chat_history" variable should contain a []core.Message.
func MessagesPlaceholder(variableName string) MessageTemplate {
	return Placeholder(variableName)
}
