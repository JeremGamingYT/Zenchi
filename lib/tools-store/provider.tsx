"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from "react"

const TOOLS_STORAGE_KEY = "zenchi-tools-state"

export type ThinkingLevel = "off" | "low" | "medium" | "high"

interface ToolsState {
    planningEnabled: boolean
    agentsEnabled: boolean
    thinkingLevel: ThinkingLevel
    setPlanningEnabled: (enabled: boolean) => void
    setAgentsEnabled: (enabled: boolean) => void
    setThinkingLevel: (level: ThinkingLevel) => void
    togglePlanning: () => void
    toggleAgents: () => void
}

const ToolsContext = createContext<ToolsState | undefined>(undefined)

function getStoredState(): { planningEnabled: boolean; agentsEnabled: boolean; thinkingLevel: ThinkingLevel } {
    if (typeof window === "undefined") return { planningEnabled: false, agentsEnabled: false, thinkingLevel: "off" }

    try {
        const stored = localStorage.getItem(TOOLS_STORAGE_KEY)
        if (stored) {
            const parsed = JSON.parse(stored)
            return {
                planningEnabled: parsed.planningEnabled ?? false,
                agentsEnabled: parsed.agentsEnabled ?? false,
                thinkingLevel: parsed.thinkingLevel ?? "off"
            }
        }
    } catch {
        // Ignore parsing errors
    }
    return { planningEnabled: false, agentsEnabled: false, thinkingLevel: "off" }
}

export function ToolsProvider({ children }: { children: ReactNode }) {
    const [planningEnabled, setPlanningEnabled] = useState(false)
    const [agentsEnabled, setAgentsEnabled] = useState(false)
    const [thinkingLevel, setThinkingLevel] = useState<ThinkingLevel>("off")
    const [isInitialized, setIsInitialized] = useState(false)

    // Load from localStorage on mount
    useEffect(() => {
        const stored = getStoredState()
        setPlanningEnabled(stored.planningEnabled)
        setAgentsEnabled(stored.agentsEnabled)
        setThinkingLevel(stored.thinkingLevel)
        setIsInitialized(true)
    }, [])

    // Save to localStorage on change
    useEffect(() => {
        if (!isInitialized) return

        localStorage.setItem(
            TOOLS_STORAGE_KEY,
            JSON.stringify({ planningEnabled, agentsEnabled, thinkingLevel })
        )
    }, [planningEnabled, agentsEnabled, thinkingLevel, isInitialized])

    const togglePlanning = () => setPlanningEnabled((prev) => !prev)
    const toggleAgents = () => setAgentsEnabled((prev) => !prev)

    return (
        <ToolsContext.Provider
            value={{
                planningEnabled,
                agentsEnabled,
                thinkingLevel,
                setPlanningEnabled,
                setAgentsEnabled,
                setThinkingLevel,
                togglePlanning,
                toggleAgents,
            }}
        >
            {children}
        </ToolsContext.Provider>
    )
}

export function useTools() {
    const context = useContext(ToolsContext)
    if (!context) {
        throw new Error("useTools must be used within ToolsProvider")
    }
    return context
}
