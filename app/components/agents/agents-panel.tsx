"use client"

import { useAgents, AgentType, AGENT_CONFIGS } from "@/lib/agents-store/provider"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { X, CircleNotch, ArrowsClockwise } from "@phosphor-icons/react"

interface AgentsPanelProps {
    isOpen: boolean
    onClose: () => void
}

function AgentCard({ agentType, isActive }: { agentType: AgentType; isActive: boolean }) {
    const { toggleAgent, getAgentConfig } = useAgents()
    const config = getAgentConfig(agentType)
    const isOrchestrator = agentType === "orchestrator"

    return (
        <motion.button
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            onClick={() => !isOrchestrator && toggleAgent(agentType)}
            className={cn(
                "flex items-center gap-3 p-3 rounded-lg border transition-all",
                "hover:shadow-md cursor-pointer",
                isActive
                    ? "border-primary bg-primary/5 ring-1 ring-primary/30"
                    : "border-border bg-background/50 hover:bg-muted/50",
                isOrchestrator && "cursor-default ring-2 ring-primary/50"
            )}
            style={{
                borderLeftColor: isActive ? config.color : undefined,
                borderLeftWidth: isActive ? "3px" : undefined,
            }}
        >
            <span className="text-2xl">{config.icon}</span>
            <div className="flex-1 text-left">
                <div className="font-medium text-sm flex items-center gap-2">
                    {config.name}
                    {isOrchestrator && (
                        <span className="text-[10px] bg-primary/20 text-primary px-1.5 py-0.5 rounded">
                            CORE
                        </span>
                    )}
                </div>
                <div className="text-xs text-muted-foreground line-clamp-1">
                    {config.description}
                </div>
            </div>
            {isActive && !isOrchestrator && (
                <div
                    className="size-3 rounded-full"
                    style={{ backgroundColor: config.color }}
                />
            )}
        </motion.button>
    )
}

function AgentFlow() {
    const { state } = useAgents()
    const activeConfigs = state.activeAgents.map((id) => AGENT_CONFIGS[id])

    if (activeConfigs.length <= 1) {
        return (
            <div className="text-center text-muted-foreground text-sm py-4">
                Enable more agents to see the workflow
            </div>
        )
    }

    return (
        <div className="flex items-center justify-center gap-2 py-4 flex-wrap">
            {activeConfigs.map((config, index) => (
                <div key={config.id} className="flex items-center gap-2">
                    <div
                        className="flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium"
                        style={{
                            backgroundColor: `${config.color}20`,
                            color: config.color,
                        }}
                    >
                        <span>{config.icon}</span>
                        <span>{config.name}</span>
                    </div>
                    {index < activeConfigs.length - 1 && (
                        <ArrowsClockwise className="size-4 text-muted-foreground" />
                    )}
                </div>
            ))}
        </div>
    )
}

function MemoryStatus() {
    const { state, clearMemory } = useAgents()
    const { memoryBank } = state

    const totalMemory =
        memoryBank.episodic.length +
        memoryBank.semantic.length +
        memoryBank.working.length +
        memoryBank.validation.length

    return (
        <div className="border rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Shared Memory</span>
                {totalMemory > 0 && (
                    <button
                        onClick={clearMemory}
                        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                        Clear
                    </button>
                )}
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center justify-between px-2 py-1 bg-muted/50 rounded">
                    <span>Episodic</span>
                    <span className="text-muted-foreground">{memoryBank.episodic.length}</span>
                </div>
                <div className="flex items-center justify-between px-2 py-1 bg-muted/50 rounded">
                    <span>Semantic</span>
                    <span className="text-muted-foreground">{memoryBank.semantic.length}</span>
                </div>
                <div className="flex items-center justify-between px-2 py-1 bg-muted/50 rounded">
                    <span>Working</span>
                    <span className="text-muted-foreground">{memoryBank.working.length}</span>
                </div>
                <div className="flex items-center justify-between px-2 py-1 bg-muted/50 rounded">
                    <span>Validation</span>
                    <span className="text-muted-foreground">{memoryBank.validation.length}</span>
                </div>
            </div>
        </div>
    )
}

export function AgentsPanel({ isOpen, onClose }: AgentsPanelProps) {
    const { state, setEnabled } = useAgents()
    const agentTypes: AgentType[] = [
        "orchestrator",
        "research",
        "design",
        "backend",
        "frontend",
        "data",
        "validation",
        "review",
        "debugger",
    ]

    return (
        <AnimatePresence>
            {isOpen && (
                <motion.div
                    initial={{ opacity: 0, x: 300 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 300 }}
                    transition={{ type: "spring", damping: 25, stiffness: 300 }}
                    className="fixed right-0 top-0 h-full w-80 bg-background border-l shadow-xl z-50 flex flex-col"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between p-4 border-b">
                        <div>
                            <h2 className="font-semibold">Agents Mode</h2>
                            <p className="text-xs text-muted-foreground">Multi-agent orchestration</p>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-1.5 hover:bg-muted rounded-md transition-colors"
                        >
                            <X className="size-4" />
                        </button>
                    </div>

                    {/* Status */}
                    <div className="p-4 border-b">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium">
                                {state.enabled ? "Active" : "Disabled"}
                            </span>
                            <div
                                className={cn(
                                    "size-2 rounded-full",
                                    state.enabled ? "bg-green-500 animate-pulse" : "bg-muted"
                                )}
                            />
                        </div>
                        <p className="text-xs text-muted-foreground">
                            {state.activeAgents.length} agent{state.activeAgents.length !== 1 ? "s" : ""} configured
                        </p>
                    </div>

                    {/* Agent Flow Visualization */}
                    <div className="px-4 py-2 border-b">
                        <AgentFlow />
                    </div>

                    {/* Agent List */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-2">
                        <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
                            Available Agents
                        </h3>
                        {agentTypes.map((type) => (
                            <AgentCard
                                key={type}
                                agentType={type}
                                isActive={state.activeAgents.includes(type)}
                            />
                        ))}
                    </div>

                    {/* Memory Status */}
                    <div className="p-4 border-t">
                        <MemoryStatus />
                    </div>

                    {/* Current Task */}
                    {state.currentTask && (
                        <div className="p-4 border-t">
                            <div className="flex items-center gap-2 text-sm">
                                <CircleNotch className="size-4 animate-spin text-primary" />
                                <span className="font-medium">Current Task</span>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {state.currentTask.title}
                            </p>
                        </div>
                    )}
                </motion.div>
            )}
        </AnimatePresence>
    )
}
