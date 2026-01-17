import { generateText } from "ai"
import { createClient } from "@supabase/supabase-js"
import { Database } from "@/app/types/database.types"

export async function generateChatTitle(
    userMessage: string,
    model: any
): Promise<string> {
    try {
        const { text } = await generateText({
            model,
            system: "You are a helpful assistant. You generate very short, concise titles for conversations.",
            prompt: `Generate a title for this conversation based on the first user message. 
The title should be EXTREMELY short (3-5 words max). 
Do not use quotes.
User Message: ${userMessage.substring(0, 1000)}`, // Truncate to avoid context limit issues
        })

        return text.trim().replace(/^["']|["']$/g, "") // Remove quotes if any
    } catch (error) {
        console.error("Failed to generate title:", error)
        return "New Conversation"
    }
}

export async function updateChatTitle(
    userId: string,
    chatId: string,
    title: string,
    supabase: ReturnType<typeof createClient<Database>>
) {
    if (!supabase) return

    try {
        await supabase
            .from("chats")
            .update({ title })
            .eq("id", chatId)
            .eq("user_id", userId)
    } catch (error) {
        console.error("Failed to update chat title in DB:", error)
    }
}
