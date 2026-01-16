"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from "react"

const TOOLS_STORAGE_KEY = "zenchi-tools-state"

interface ToolsState {
    planningEnabled: boolean
    agentsEnabled: boolean
    setPlanningEnabled: (enabled: boolean) => void
    setAgentsEnabled: (enabled: boolean) => void
    togglePlanning: () => void
    toggleAgents: () => void
}

const ToolsContext = createContext<ToolsState | undefined>(undefined)

function getStoredState(): { planningEnabled: boolean; agentsEnabled: boolean } {
    if (typeof window === "undefined") return { planningEnabled: false, agentsEnabled: false }

    try {
        const stored = localStorage.getItem(TOOLS_STORAGE_KEY)
        if (stored) {
            return JSON.parse(stored)
        }
    } catch {
        // Ignore parsing errors
    }
    return { planningEnabled: false, agentsEnabled: false }
}

export function ToolsProvider({ children }: { children: ReactNode }) {
    const [planningEnabled, setPlanningEnabled] = useState(false)
    const [agentsEnabled, setAgentsEnabled] = useState(false)
    const [isInitialized, setIsInitialized] = useState(false)

    // Load from localStorage on mount
    useEffect(() => {
        const stored = getStoredState()
        setPlanningEnabled(stored.planningEnabled)
        setAgentsEnabled(stored.agentsEnabled)
        setIsInitialized(true)
    }, [])

    // Save to localStorage on change
    useEffect(() => {
        if (!isInitialized) return

        localStorage.setItem(
            TOOLS_STORAGE_KEY,
            JSON.stringify({ planningEnabled, agentsEnabled })
        )
    }, [planningEnabled, agentsEnabled, isInitialized])

    const togglePlanning = () => setPlanningEnabled((prev) => !prev)
    const toggleAgents = () => setAgentsEnabled((prev) => !prev)

    return (
        <ToolsContext.Provider
            value={{
                planningEnabled,
                agentsEnabled,
                setPlanningEnabled,
                setAgentsEnabled,
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
