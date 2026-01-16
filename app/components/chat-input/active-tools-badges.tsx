"use client"

import { useTools } from "@/lib/tools-store/provider"
import { Kanban, Robot } from "@phosphor-icons/react"
import { motion, AnimatePresence } from "framer-motion"

export function ActiveToolsBadges() {
    const { planningEnabled, agentsEnabled } = useTools()

    if (!planningEnabled && !agentsEnabled) {
        return null
    }

    return (
        <div className="flex gap-1">
            <AnimatePresence>
                {planningEnabled && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="flex items-center gap-1 rounded-md bg-primary/10 px-1.5 py-0.5 text-xs text-primary"
                    >
                        <Kanban className="size-3" weight="bold" />
                        <span>Planning</span>
                    </motion.div>
                )}
                {agentsEnabled && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        className="flex items-center gap-1 rounded-md bg-primary/10 px-1.5 py-0.5 text-xs text-primary"
                    >
                        <Robot className="size-3" weight="bold" />
                        <span>Agents</span>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
