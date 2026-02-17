"use client"

import { motion, AnimatePresence } from "framer-motion"
import { ChatText, Trash, X } from "@phosphor-icons/react"

interface CommentBubbleProps {
    isVisible: boolean
    position: { x: number; y: number }
    comment: string
    onDelete: () => void
    onClose: () => void
}

export function CommentBubble({
    isVisible,
    position,
    comment,
    onDelete,
    onClose
}: CommentBubbleProps) {
    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.9 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.9 }}
                    transition={{ type: "spring", bounce: 0.3, duration: 0.35 }}
                    style={{
                        position: 'fixed',
                        left: position.x,
                        top: position.y,
                        transform: "translateX(-50%)"
                    }}
                    className="z-[105] w-72 origin-top"
                >
                    <div className="rounded-xl bg-violet-950/95 backdrop-blur-xl border border-violet-500/20 shadow-xl shadow-violet-900/30 overflow-hidden">
                        {/* Header */}
                        <div className="px-3 py-2 border-b border-violet-500/20 flex items-center justify-between bg-violet-500/10">
                            <div className="flex items-center gap-2">
                                <ChatText size={14} className="text-violet-400" weight="fill" />
                                <span className="text-xs font-semibold text-violet-300">AI Comment</span>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-1 rounded-md hover:bg-white/10 transition-colors text-white/50 hover:text-white"
                            >
                                <X size={12} />
                            </button>
                        </div>

                        {/* Comment Content */}
                        <div className="p-3">
                            <p className="text-sm text-white/80 leading-relaxed">{comment}</p>
                        </div>

                        {/* Footer */}
                        <div className="px-3 pb-3 flex justify-end">
                            <motion.button
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={onDelete}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-red-400/80 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                            >
                                <Trash size={12} />
                                Remove
                            </motion.button>
                        </div>
                    </div>

                    {/* Arrow pointer */}
                    <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-0 h-0 border-l-8 border-r-8 border-b-8 border-transparent border-b-violet-950/95" />
                </motion.div>
            )}
        </AnimatePresence>
    )
}
