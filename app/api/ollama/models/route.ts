import { NextRequest, NextResponse } from "next/server"
import { fetchOllamaModels } from "@/lib/models/ollama"

// Endpoint to fetch Ollama models from configured endpoint
export async function POST(request: NextRequest) {
    try {
        const body = await request.json()
        const { endpoint } = body

        if (!endpoint) {
            return NextResponse.json(
                { error: "No Ollama endpoint provided", models: [] },
                { status: 400 }
            )
        }

        const models = await fetchOllamaModels(endpoint)

        return NextResponse.json({
            models,
            endpoint,
            timestamp: new Date().toISOString(),
        })
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Unknown error"
        console.error("Error fetching Ollama models:", errorMessage)
        return NextResponse.json(
            { error: `Failed to fetch Ollama models: ${errorMessage}`, models: [] },
            { status: 500 }
        )
    }
}
