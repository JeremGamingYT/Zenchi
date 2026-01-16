import { ModelConfig } from "./types"

// Fetch models from Ollama endpoint
export async function fetchOllamaModels(endpoint: string): Promise<ModelConfig[]> {
    try {
        // Detect if this is an NGrok URL
        const isNgrok = endpoint.includes("ngrok")

        // Build headers - add NGrok bypass header if needed
        const headers: HeadersInit = {
            "Content-Type": "application/json",
        }

        if (isNgrok) {
            headers["ngrok-skip-browser-warning"] = "true"
        }

        const response = await fetch(`${endpoint}/api/tags`, {
            method: "GET",
            headers,
        })

        if (!response.ok) {
            console.warn(`Failed to fetch Ollama models from ${endpoint}: ${response.status}`)
            return []
        }

        const data = await response.json()
        const models: ModelConfig[] = (data.models || []).map((model: { name: string; details?: { parameter_size?: string; family?: string } }) => ({
            id: model.name,
            name: formatModelName(model.name),
            providerId: "ollama",
            providerName: "Ollama",
            uniqueId: `ollama/${model.name}`,
            accessible: true,
            contextLength: 4096, // Default, Ollama doesn't expose this
            description: model.details?.parameter_size
                ? `${model.details.parameter_size} parameters`
                : "Local Ollama model",
            vision: false,
            maxOutput: 4096,
            inputPerMillion: 0, // Free local model
            outputPerMillion: 0,
            tools: true, // Ollama supports tool calling
            // Store endpoint in special field for later use
            ollamaEndpoint: endpoint,
            // apiSdk will be created dynamically when using the model
            apiSdk: async (_apiKey?: string, _opts?: { enableSearch?: boolean }) => {
                const { createOllamaModel } = await import("./ollama-sdk")
                return createOllamaModel(endpoint, model.name)
            }
        }))

        return models
    } catch (error) {
        console.warn(`Error fetching Ollama models: ${error}`)
        return []
    }
}

function formatModelName(name: string): string {
    // Convert "gpt-oss-20b:latest" to "GPT-OSS 20B"
    return name
        .replace(/:latest$/, "")
        .replace(/:.*$/, "") // Remove any tag
        .split("-")
        .map(part => {
            // Handle size suffixes like "20b", "7b"
            if (/^\d+[bm]$/i.test(part)) {
                return part.toUpperCase()
            }
            // Capitalize normal words
            return part.charAt(0).toUpperCase() + part.slice(1)
        })
        .join(" ")
}
