import { NextRequest, NextResponse } from "next/server"

const OLLAMA_STORAGE_KEY = "zenchi-ollama-settings"

// Server-side proxy for Ollama API to bypass CORS and add NGrok headers
export async function POST(request: NextRequest) {
    try {
        const body = await request.json()
        const { endpoint, path = "/api/tags" } = body

        if (!endpoint) {
            return NextResponse.json(
                { error: "No endpoint provided" },
                { status: 400 }
            )
        }

        // Detect if this is an NGrok URL
        const isNgrok = endpoint.includes("ngrok")

        // Build headers - add NGrok bypass header if needed
        const headers: HeadersInit = {
            "Content-Type": "application/json",
        }

        if (isNgrok) {
            headers["ngrok-skip-browser-warning"] = "true"
        }

        // Forward request to Ollama
        const ollamaUrl = `${endpoint}${path}`
        const response = await fetch(ollamaUrl, {
            method: "GET",
            headers,
        })

        if (!response.ok) {
            return NextResponse.json(
                { error: `Ollama returned ${response.status}: ${response.statusText}` },
                { status: response.status }
            )
        }

        const data = await response.json()
        return NextResponse.json(data)
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Unknown error"
        return NextResponse.json(
            { error: `Connection failed: ${errorMessage}` },
            { status: 500 }
        )
    }
}
