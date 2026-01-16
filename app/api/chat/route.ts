import { SYSTEM_PROMPT_DEFAULT } from "@/lib/config"
import { getAllModels } from "@/lib/models"
import type { ProviderWithoutOllama } from "@/lib/user-keys"
import { getUserKey } from "@/lib/user-keys"
import { streamText, ToolSet, stepCountIs, convertToModelMessages, type UIMessage } from "ai";
import { buildMcpTools, type MCPServerConfig } from "@/lib/mcp/tools"
import type { Message } from "@/app/types/api.types"
import {
  incrementMessageCount,
  logUserMessage,
  storeAssistantMessage,
  validateAndTrackUsage,
} from "./api"
import { createErrorResponse, extractErrorMessage } from "./utils"
import { trackModelUsage } from "./usage-tracking"

export const maxDuration = 60

type ChatRequest = {
  messages: UIMessage[]
  chatId: string
  userId: string
  model: string
  isAuthenticated: boolean
  systemPrompt: string
  enableSearch: boolean
  message_group_id?: string
  mcpServers?: MCPServerConfig[]
  ollamaEndpoint?: string // For Ollama models, pass the configured endpoint
}

export async function POST(req: Request) {
  try {
    const body = await req.json()
    const {
      messages,
      chatId,
      userId,
      model,
      isAuthenticated,
      systemPrompt,
      enableSearch,
      message_group_id,
      mcpServers,
      ollamaEndpoint,
    } = body as ChatRequest

    if (!messages || !chatId || !userId) {
      return new Response(
        JSON.stringify({ error: "Error, missing information" }),
        { status: 400 }
      )
    }

    const supabase = await validateAndTrackUsage({
      userId,
      model,
      isAuthenticated,
    })

    if (supabase) {
      await incrementMessageCount({ supabase, userId })
    }

    const userMessage = messages[messages.length - 1]

    if (supabase && userMessage?.role === "user") {
      await logUserMessage({
        supabase,
        userId,
        chatId,
        parts: userMessage.parts || [],
        model,
        isAuthenticated,
        message_group_id,
      })
    }

    // Check if this is an Ollama model request
    const isOllamaModel = model.startsWith("ollama/")

    // Debug logging for Ollama
    if (isOllamaModel) {
      console.log("[Ollama Debug] Model:", model)
      console.log("[Ollama Debug] Endpoint:", ollamaEndpoint)
    }

    let modelConfig: any
    let makeModel: any

    if (isOllamaModel && ollamaEndpoint) {
      // For Ollama models, create the SDK directly using the provided endpoint
      const ollamaModelId = model.replace("ollama/", "")
      console.log("[Ollama Debug] Model ID:", ollamaModelId)

      const { createOllamaModel } = await import("@/lib/models/ollama-sdk")

      modelConfig = {
        id: ollamaModelId,
        uniqueId: model,
        providerId: "ollama",
        tools: true,
        inputCost: 0,
        outputCost: 0,
      }

      makeModel = async () => {
        console.log("[Ollama Debug] Creating model with endpoint:", ollamaEndpoint)
        return createOllamaModel(ollamaEndpoint, ollamaModelId)
      }
    } else if (isOllamaModel && !ollamaEndpoint) {
      // Ollama model but no endpoint - this is an error
      console.error("[Ollama Error] Model selected but no endpoint provided")
      throw new Error("Ollama model selected but no endpoint configured. Please configure Ollama in Settings > Connections.")
    } else {
      // Standard model handling
      const { getCustomModels } = await import("@/lib/models/custom")
      const customModels = await getCustomModels()

      // Only pass customModels to getAllModels if non-empty to avoid polluting shared cache
      const customModelsForCache = customModels && customModels.length > 0 ? customModels : undefined

      // Get global models from cache (without custom models)
      const globalModels = await getAllModels(undefined)

      // Merge custom models locally after fetching cached global models
      const allModels = customModelsForCache
        ? [...globalModels, ...customModelsForCache]
        : globalModels

      modelConfig = allModels.find((m) => m.uniqueId === model)

      if (!modelConfig || !modelConfig.apiSdk) {
        throw new Error(`Model ${model} not found`)
      }

      makeModel = modelConfig.apiSdk
    }

    const effectiveSystemPrompt = systemPrompt || SYSTEM_PROMPT_DEFAULT

    let apiKey: string | undefined
    if (isAuthenticated && userId && modelConfig.providerId !== "ollama") {
      const { getEffectiveApiKey } = await import("@/lib/user-keys")
      let key = await getEffectiveApiKey(userId, modelConfig.providerId as ProviderWithoutOllama)
      if (!key) {
        try {
          key = await getUserKey(userId, modelConfig.providerId as any)
        } catch { }
      }
      apiKey = key || undefined
    }

    if (!makeModel) {
      throw new Error(`Selected model ${model} is not invokable`)
    }

    if (!Array.isArray(messages)) {
      return new Response(
        JSON.stringify({ error: "Invalid messages format" }),
        { status: 400 }
      )
    }

    const modelMessages = convertToModelMessages(messages)

    // Load MCP tools from user's configured servers (or env vars as fallback)
    // Validate mcpServers is an array before filtering
    const enabledMcpServers = Array.isArray(mcpServers)
      ? mcpServers.filter(s => s && typeof s === 'object' && s.enabled === true)
      : []
    const { tools: mcpTools, close: closeMcp } = await buildMcpTools(enabledMcpServers)

    // Ensure closeMcp is invoked exactly once
    let mcpClosed = false
    const safeCloseMcp = async () => {
      if (mcpClosed || !closeMcp) return
      mcpClosed = true
      try {
        await closeMcp()
      } catch (closeError) {
        console.error("Error closing MCP transports:", closeError)
      }
    }

    try {
      const modelInstance = await makeModel(apiKey, { enableSearch })

      // Add local tools
      const { localTools } = await import("@/lib/tools/local")

      const streamTextOptions: Parameters<typeof streamText>[0] = {
        model: modelInstance,
        system: effectiveSystemPrompt,
        messages: modelMessages,
        // @ts-ignore - maxSteps is supported in recent ai SDK versions
        maxSteps: 10, // Enable multi-turn tool calling (up to 10 steps)
        tools: {
          ...localTools,
          ...(mcpTools || {})
        } as any, // Cast to any to avoid complex tool type mismatches
        stopWhen: stepCountIs(10),
        onFinish: async ({ response, usage }) => {
          try {
            let savedMessageId: number | undefined

            if (supabase && response.messages?.length) {
              // Only use response.messages - it contains the complete final response
              // Steps are intermediate and already included in the final response
              await storeAssistantMessage({
                supabase,
                chatId,
                messages: response.messages as any[],
                message_group_id,
                model,
              })

              // Try to get the message ID for tracking (query the most recent assistant message for this chat)
              try {
                const { data: messageData } = await supabase
                  .from("messages")
                  .select("id")
                  .eq("chat_id", chatId)
                  .eq("role", "assistant")
                  .order("created_at", { ascending: false })
                  .limit(1)
                  .maybeSingle()

                if (messageData) {
                  savedMessageId = messageData.id
                }
              } catch (err) {
                console.error("Error fetching message ID:", err)
              }
            }

            // Track model usage if we have token data and pricing
            if (supabase && usage && (usage.inputTokens || usage.outputTokens)) {
              await trackModelUsage({
                supabase,
                userId,
                chatId,
                messageId: savedMessageId,
                modelId: modelConfig.id,
                providerId: modelConfig.providerId,
                usage: {
                  inputTokens: usage.inputTokens || 0,
                  outputTokens: usage.outputTokens || 0,
                  totalTokens: usage.totalTokens || 0,
                },
                pricing: {
                  inputCost: modelConfig.inputCost,
                  outputCost: modelConfig.outputCost,
                },
              })
            }
          } catch (saveError) {
            console.error("Error in onFinish:", saveError)
          } finally {
            await safeCloseMcp()
          }
        },
        onError: async (error: unknown) => {
          await safeCloseMcp()
          throw error
        }
      }



      const result = streamText(streamTextOptions)

      return result.toUIMessageStreamResponse({
        sendReasoning: true,
        sendSources: true,
        messageMetadata: ({ part }) => {
          if (part.type === "finish") {
            return { totalUsage: part.totalUsage }
          }
        },
        onError: (error: unknown) => {
          // Close MCP without blocking (fire and forget)
          safeCloseMcp().catch(e => console.error("Error closing MCP in onError:", e))
          return extractErrorMessage(error)
        },
      })
    } catch (streamError) {
      await safeCloseMcp()
      throw streamError
    }
  } catch (err: unknown) {
    // Don't log BudgetExceededError to console (user-facing, not a system error)
    const errorName = (err as any)?.name

    if (errorName !== "BudgetExceededError") {
      console.error("Error in /api/chat:", err)
      // Extra debug for Ollama
      if (err instanceof Error) {
        console.error("[Ollama Debug] Error name:", err.name)
        console.error("[Ollama Debug] Error message:", err.message)
        console.error("[Ollama Debug] Error stack:", err.stack)
      }
    }

    const error = err as {
      code?: string
      message?: string
      statusCode?: number
      name?: string
    }

    return new Response(JSON.stringify({
      error: error.message || "Internal server error",
      debug_error: {
        name: (err as any)?.name,
        message: (err as any)?.message,
        stack: (err as any)?.stack,
        raw: String(err)
      }
    }), { status: 200 })
  }
}
