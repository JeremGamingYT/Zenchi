"use client"

import { Brain, ArrowLeft, Check } from "@phosphor-icons/react"
import React from "react"
import { motion } from "framer-motion"
import {
    DropdownMenuItem,
    DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { useTools, ThinkingLevel } from "@/lib/tools-store/provider"

type SubmenuThinkingProps = {
    onBack: () => void
}

const THINKING_LEVELS: { id: ThinkingLevel; label: string; description?: string }[] = [
    { id: "off", label: "Disabled" },
    { id: "low", label: "Low Reasoning" },
    { id: "medium", label: "Medium Reasoning" },
    { id: "high", label: "High Reasoning" },
]

export function SubmenuThinking({ onBack }: SubmenuThinkingProps) {
    const { thinkingLevel, setThinkingLevel } = useTools()

    return (
        <motion.div
            key="thinking-submenu"
            initial={{ x: 320, opacity: 1 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 320, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="flex flex-col w-full p-1.5"
        >
            <DropdownMenuItem
                onClick={(e) => {
                    e.preventDefault()
                    onBack()
                }}
                onSelect={(e) => e.preventDefault()}
                className="gap-2.5 h-8 cursor-pointer"
            >
                <ArrowLeft className="size-4 opacity-50" />
                <span className="opacity-60">Thinking Level</span>
            </DropdownMenuItem>

            {THINKING_LEVELS.map((level) => (
                <DropdownMenuItem
                    key={level.id}
                    onClick={(e) => {
                        e.preventDefault()
                        setThinkingLevel(level.id)
                    }}
                    onSelect={(e) => e.preventDefault()}
                    className="gap-2.5 h-8 cursor-pointer"
                >
                    <Brain className="size-4" />
                    <span>{level.label}</span>
                    {thinkingLevel === level.id && (
                        <Check className="ml-auto size-4 text-primary" weight="bold" />
                    )}
                </DropdownMenuItem>
            ))}
        </motion.div>
    )
}
