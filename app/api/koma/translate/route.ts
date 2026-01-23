import { createGoogleGenerativeAI } from "@ai-sdk/google"
import { generateText } from "ai"
import { getUserKey, getEffectiveApiKey } from "@/lib/user-keys"
import { createClient } from "@/lib/supabase/server"
import { isSupabaseEnabled } from "@/lib/supabase/config"

export const maxDuration = 120

type TranslateRequest = {
    text: string
    sourceLanguage: string
    targetLanguage: string
    context?: string // Additional context for better translation (e.g., "anime dialogue", "webtoon speech bubble")
    userId?: string
    isAuthenticated?: boolean
}

const TRANSLATION_SYSTEM_PROMPT = `You are an expert translator specialized in anime, webtoon, and manga content translation.

Your task is to translate text while:
1. Preserving the original meaning, tone, and emotional nuance
2. Adapting cultural references appropriately for the target audience
3. Maintaining character voice consistency
4. Handling honorifics, sound effects, and expressions naturally
5. Keeping the translation concise for speech bubbles/subtitles

IMPORTANT RULES:
- Return ONLY the translated text, no explanations
- Preserve any formatting, line breaks, or special characters
- If the source contains sound effects (onomatopoeia), translate them naturally or keep them if commonly understood
- Maintain the register (formal/informal) of the original

Context helps improve translation quality. Use it when provided.`

export async function POST(req: Request) {
    try {
        const body = await req.json()
        const {
            text,
            sourceLanguage,
            targetLanguage,
            context,
            userId,
            isAuthenticated,
        } = body as TranslateRequest

        if (!text || !sourceLanguage || !targetLanguage) {
            return new Response(
                JSON.stringify({ error: "Missing required fields: text, sourceLanguage, targetLanguage" }),
                { status: 400 }
            )
        }

        // Get API key
        let apiKey: string | undefined

        if (isAuthenticated && userId) {
            try {
                apiKey = await getEffectiveApiKey(userId, "google")
                if (!apiKey) {
                    apiKey = await getUserKey(userId, "google")
                }
            } catch (e) {
                // Fall through to env key
            }
        }

        // Fallback to environment variable
        if (!apiKey) {
            apiKey = process.env.GOOGLE_GENERATIVE_AI_API_KEY
        }

        if (!apiKey) {
            return new Response(
                JSON.stringify({ error: "No Google API key configured. Please add your API key in Settings." }),
                { status: 401 }
            )
        }

        // Create Gemini model
        const google = createGoogleGenerativeAI({ apiKey })
        const model = google("gemini-2.5-pro")

        // Build the translation prompt
        let userPrompt = `Translate the following text from ${sourceLanguage} to ${targetLanguage}.`

        if (context) {
            userPrompt += `\n\nContext: ${context}`
        }

        userPrompt += `\n\nText to translate:\n${text}`

        // Generate translation
        const result = await generateText({
            model,
            system: TRANSLATION_SYSTEM_PROMPT,
            prompt: userPrompt,
        })

        return new Response(
            JSON.stringify({
                translation: result.text,
                usage: {
                    inputTokens: result.usage?.inputTokens || 0,
                    outputTokens: result.usage?.outputTokens || 0,
                },
            }),
            { status: 200, headers: { "Content-Type": "application/json" } }
        )
    } catch (err: unknown) {
        console.error("Error in /api/koma/translate:", err)

        const error = err as { message?: string }
        return new Response(
            JSON.stringify({ error: error.message || "Translation failed" }),
            { status: 500 }
        )
    }
}
