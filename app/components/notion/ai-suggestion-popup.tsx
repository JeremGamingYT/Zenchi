"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Check, X, ArrowsClockwise } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"

interface AISuggestionPopupProps {
    isVisible: boolean
    position: { x: number; y: number }
    originalText: string
    suggestedText: string
    type: 'grammar' | 'style' | 'improve'
    onAccept: () => void
    onReject: () => void
}

const TYPE_CONFIG = {
    grammar: {
        label: "Grammar Fix",
        color: "text-green-400",
        bgAccent: "from-green-500/20",
    },
    style: {
        label: "Style Improvement",
        color: "text-blue-400",
        bgAccent: "from-blue-500/20",
    },
    improve: {
        label: "AI Improvement",
        color: "text-amber-400",
        bgAccent: "from-amber-500/20",
    },
}

export function AISuggestionPopup({
    isVisible,
    position,
    originalText,
    suggestedText,
    type,
    onAccept,
    onReject
}: AISuggestionPopupProps) {
    const config = TYPE_CONFIG[type]

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.95 }}
                    transition={{ type: "spring", bounce: 0.25, duration: 0.4 }}
                    style={{
                        position: 'fixed',
                        left: position.x,
                        top: position.y,
                        transform: "translateX(-50%)"
                    }}
                    className="z-[110] w-80 origin-top"
                >
                    <div className={cn(
                        "rounded-xl bg-black/95 backdrop-blur-xl border border-white/10 shadow-2xl overflow-hidden",
                    )}>
                        {/* Header */}
                        <div className={cn(
                            "px-4 py-2.5 border-b border-white/10 bg-gradient-to-r to-transparent",
                            config.bgAccent
                        )}>
                            <div className="flex items-center gap-2">
                                <ArrowsClockwise size={14} className={config.color} weight="fill" />
                                <span className="text-xs font-semibold text-white/90">{config.label}</span>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="p-4 space-y-3">
                            {/* Original */}
                            <div className="space-y-1">
                                <span className="text-[10px] uppercase tracking-wider text-white/40 font-medium">Original</span>
                                <p className="text-sm text-white/60 line-through decoration-red-400/50">{originalText}</p>
                            </div>

                            {/* Suggested */}
                            <div className="space-y-1">
                                <span className="text-[10px] uppercase tracking-wider text-white/40 font-medium">Suggested</span>
                                <p className="text-sm text-white/90 font-medium">{suggestedText}</p>
                            </div>
                        </div>

                        {/* Actions */}
                        <div className="flex items-center gap-2 px-4 pb-4">
                            <motion.button
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={onAccept}
                                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-green-500/20 text-green-400 text-xs font-semibold hover:bg-green-500/30 transition-colors border border-green-500/20"
                            >
                                <Check size={14} weight="bold" />
                                Accept
                            </motion.button>
                            <motion.button
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={onReject}
                                className="flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-white/5 text-white/60 text-xs font-semibold hover:bg-white/10 transition-colors border border-white/10"
                            >
                                <X size={14} weight="bold" />
                                Dismiss
                            </motion.button>
                        </div>
                    </div>

                    {/* Glow */}
                    <div className={cn(
                        "absolute inset-0 -z-10 rounded-xl blur-xl opacity-30",
                        type === 'grammar' && "bg-green-500/30",
                        type === 'style' && "bg-blue-500/30",
                        type === 'improve' && "bg-amber-500/30",
                    )} />
                </motion.div>
            )}
        </AnimatePresence>
    )
}
