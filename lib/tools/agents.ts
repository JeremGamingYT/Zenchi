import { tool } from "ai"
import { z } from "zod"
import { AGENT_PROMPTS, type AgentRole } from "@/lib/agents/registry"
import { createOllamaModel } from "@/lib/models/ollama-sdk"
import { generateText } from "ai"

// Define schema outside to ensure type inference works
const consultExpertSchema = z.object({
    role: z.enum([
        "expert_researcher",
        "expert_architect",
        "expert_implementer",
        "expert_critic",
        "expert_tester"
    ]).describe("The role of the expert to consult."),
    task: z.string().describe("The specific task or question for the expert. Be detailed."),
    context: z.string().optional().describe("Relevant context, file contents, or previous findings to pass to the expert.")
})

type ConsultExpertArgs = z.infer<typeof consultExpertSchema>

export const createAgentTools = (model: any) => {
    return {
        consult_expert: tool({
            description: "Consult a specialized expert agent for a specific task. Use this to delegate work.",
            parameters: consultExpertSchema,
            execute: async (args: ConsultExpertArgs) => {
                const { role, task, context } = args

                // Type safety for role
                if (!role || !(role in AGENT_PROMPTS)) {
                    return { error: "Invalid expert role specified." }
                }

                const systemPrompt = AGENT_PROMPTS[role as AgentRole]

                try {
                    const response = await generateText({
                        model,
                        system: systemPrompt,
                        prompt: `TASK: ${task}\n\nCONTEXT:\n${context || "No additional context provided."}`,
                    })

                    return {
                        expert: role,
                        response: response.text
                    }
                } catch (error: any) {
                    return { error: `Failed to consult expert ${role}: ${error.message}` }
                }
            }
        })
    }
}
