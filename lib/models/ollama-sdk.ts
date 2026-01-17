/**
 * Ollama SDK creation - server-side only
 * Creates an OpenAI-compatible model instance for Ollama endpoints
 */

export async function createOllamaModel(endpoint: string, modelId: string, thinkingLevel?: string) {
    // Ollama uses OpenAI-compatible API format
    const { createOpenAICompatible } = await import("@ai-sdk/openai-compatible")

    // Normalize endpoint - Ollama's OpenAI-compatible API is at /v1
    const baseURL = endpoint.replace(/\/+$/, "") + "/v1"

    // Detect if this is an NGrok URL and add headers
    const isNgrok = endpoint.includes("ngrok")

    const instance = createOpenAICompatible({
        name: "ollama",
        baseURL,
        // Ollama doesn't require an API key
        apiKey: "ollama",
        headers: isNgrok ? {
            "ngrok-skip-browser-warning": "true",
        } : undefined,
        fetch: async (url, options) => {
            if (options?.body) {
                try {
                    const body = JSON.parse(options.body as string)

                    // Handle 'think' parameter for gpt-oss and similar models
                    if (thinkingLevel && thinkingLevel !== "off") {
                        if (modelId.includes("gpt-oss")) {
                            // Specialized support for gpt-oss which expects "low"|"medium"|"high"
                            body.think = thinkingLevel
                        } else {
                            // Standard/Future Ollama support (budget based)
                            const budgets: Record<string, number> = { low: 1024, medium: 4096, high: 32768 }
                            const budget = budgets[thinkingLevel] || 0
                            if (budget > 0) {
                                body.thinking = { budget }
                            }
                        }
                    }

                    options.body = JSON.stringify(body)
                } catch (e) {
                    console.error("Failed to inject thinking params:", e)
                }
            }
            return fetch(url, options)
        }
    })

    return instance(modelId)
}
