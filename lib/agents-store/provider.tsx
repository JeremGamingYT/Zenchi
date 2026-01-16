"use client"

import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from "react"

// Agent Types based on agents-mode.md specification
export type AgentType =
    | "orchestrator"
    | "research"
    | "design"
    | "backend"
    | "frontend"
    | "data"
    | "validation"
    | "review"
    | "debugger"

export interface AgentConfig {
    id: AgentType
    name: string
    description: string
    icon: string
    color: string
    enabled: boolean
}

export interface AgentMessage {
    id: string
    from: AgentType
    to: AgentType
    context: {
        taskTree: string
        currentState: string
        constraints: string[]
        previousAttempts: string[]
    }
    payload: string
    verificationRequirements: string[]
    confidenceLevel: number
    timestamp: number
}

export interface Task {
    id: string
    title: string
    status: "pending" | "in-progress" | "completed" | "failed"
    assignedAgent: AgentType
    subtasks: Task[]
    result?: string
    confidenceLevel?: number
}

export interface AgentsState {
    enabled: boolean
    activeAgents: AgentType[]
    currentTask: Task | null
    messageHistory: AgentMessage[]
    memoryBank: {
        episodic: string[]    // Historical actions and decisions
        semantic: string[]    // Accumulated knowledge, patterns
        working: string[]     // Active context, dependencies
        validation: string[]  // Tests, verifications, metrics
    }
}

const AGENTS_STORAGE_KEY = "zenchi-agents-state"

// Agent configurations
export const AGENT_CONFIGS: Record<AgentType, Omit<AgentConfig, "enabled">> = {
    orchestrator: {
        id: "orchestrator",
        name: "Orchestrator",
        description: "Meta-cognitive coordinator that plans, delegates, and integrates results",
        icon: "🧠",
        color: "#8B5CF6",
    },
    research: {
        id: "research",
        name: "Researcher",
        description: "Accesses documentation, web, and knowledge bases with cross-referencing",
        icon: "🔍",
        color: "#3B82F6",
    },
    design: {
        id: "design",
        name: "Architect",
        description: "Designs solutions, generates specifications, validates feasibility",
        icon: "📐",
        color: "#10B981",
    },
    backend: {
        id: "backend",
        name: "Backend Specialist",
        description: "Implements server-side code, APIs, and database logic",
        icon: "⚙️",
        color: "#F59E0B",
    },
    frontend: {
        id: "frontend",
        name: "Frontend Specialist",
        description: "Implements UI components, styling, and user interactions",
        icon: "🎨",
        color: "#EC4899",
    },
    data: {
        id: "data",
        name: "Data/ML Specialist",
        description: "Handles data processing, machine learning, and analytics",
        icon: "📊",
        color: "#06B6D4",
    },
    validation: {
        id: "validation",
        name: "Validator",
        description: "Runs automated tests, static analysis, and conformity checks",
        icon: "✅",
        color: "#22C55E",
    },
    review: {
        id: "review",
        name: "Critic",
        description: "Performs code review, security analysis, and optimization suggestions",
        icon: "👁️",
        color: "#EF4444",
    },
    debugger: {
        id: "debugger",
        name: "Debugger",
        description: "Analyzes errors, formulates root cause hypotheses, monitors fixes",
        icon: "🐛",
        color: "#F97316",
    },
}

interface AgentsContextType {
    state: AgentsState
    setEnabled: (enabled: boolean) => void
    toggleAgent: (agentType: AgentType) => void
    isAgentActive: (agentType: AgentType) => boolean
    addMessage: (message: Omit<AgentMessage, "id" | "timestamp">) => void
    setCurrentTask: (task: Task | null) => void
    updateMemory: (type: keyof AgentsState["memoryBank"], content: string) => void
    clearMemory: () => void
    getAgentConfig: (agentType: AgentType) => AgentConfig
}

const AgentsContext = createContext<AgentsContextType | undefined>(undefined)

const defaultState: AgentsState = {
    enabled: false,
    activeAgents: ["orchestrator"],
    currentTask: null,
    messageHistory: [],
    memoryBank: {
        episodic: [],
        semantic: [],
        working: [],
        validation: [],
    },
}

function getStoredState(): AgentsState {
    if (typeof window === "undefined") return defaultState
    try {
        const stored = localStorage.getItem(AGENTS_STORAGE_KEY)
        if (stored) {
            const parsed = JSON.parse(stored)
            return { ...defaultState, ...parsed }
        }
    } catch {
        // Ignore
    }
    return defaultState
}

function saveState(state: AgentsState) {
    if (typeof window === "undefined") return
    // Only save essential state, not full message history
    const toSave = {
        enabled: state.enabled,
        activeAgents: state.activeAgents,
    }
    localStorage.setItem(AGENTS_STORAGE_KEY, JSON.stringify(toSave))
}

export function AgentsProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<AgentsState>(defaultState)
    const [isInitialized, setIsInitialized] = useState(false)

    useEffect(() => {
        const stored = getStoredState()
        setState(stored)
        setIsInitialized(true)
    }, [])

    useEffect(() => {
        if (!isInitialized) return
        saveState(state)
    }, [state, isInitialized])

    const setEnabled = useCallback((enabled: boolean) => {
        setState((prev) => ({ ...prev, enabled }))
    }, [])

    const toggleAgent = useCallback((agentType: AgentType) => {
        setState((prev) => {
            const isActive = prev.activeAgents.includes(agentType)
            // Orchestrator cannot be deactivated
            if (agentType === "orchestrator" && isActive) return prev

            return {
                ...prev,
                activeAgents: isActive
                    ? prev.activeAgents.filter((a) => a !== agentType)
                    : [...prev.activeAgents, agentType],
            }
        })
    }, [])

    const isAgentActive = useCallback((agentType: AgentType) => {
        return state.activeAgents.includes(agentType)
    }, [state.activeAgents])

    const addMessage = useCallback((message: Omit<AgentMessage, "id" | "timestamp">) => {
        const fullMessage: AgentMessage = {
            ...message,
            id: crypto.randomUUID(),
            timestamp: Date.now(),
        }
        setState((prev) => ({
            ...prev,
            messageHistory: [...prev.messageHistory, fullMessage].slice(-100), // Keep last 100
        }))
    }, [])

    const setCurrentTask = useCallback((task: Task | null) => {
        setState((prev) => ({ ...prev, currentTask: task }))
    }, [])

    const updateMemory = useCallback((type: keyof AgentsState["memoryBank"], content: string) => {
        setState((prev) => ({
            ...prev,
            memoryBank: {
                ...prev.memoryBank,
                [type]: [...prev.memoryBank[type], content].slice(-50), // Keep last 50
            },
        }))
    }, [])

    const clearMemory = useCallback(() => {
        setState((prev) => ({
            ...prev,
            memoryBank: defaultState.memoryBank,
            messageHistory: [],
            currentTask: null,
        }))
    }, [])

    const getAgentConfig = useCallback((agentType: AgentType): AgentConfig => {
        return {
            ...AGENT_CONFIGS[agentType],
            enabled: state.activeAgents.includes(agentType),
        }
    }, [state.activeAgents])

    return (
        <AgentsContext.Provider
            value={{
                state,
                setEnabled,
                toggleAgent,
                isAgentActive,
                addMessage,
                setCurrentTask,
                updateMemory,
                clearMemory,
                getAgentConfig,
            }}
        >
            {children}
        </AgentsContext.Provider>
    )
}

export function useAgents() {
    const context = useContext(AgentsContext)
    if (!context) {
        throw new Error("useAgents must be used within AgentsProvider")
    }
    return context
}
