/**
 * Ollama SDK creation - server-side only
 * Creates an OpenAI-compatible model instance for Ollama endpoints
 */

export async function createOllamaModel(endpoint: string, modelId: string) {
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
    })

    return instance(modelId)
}
