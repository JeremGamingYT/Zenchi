"use client"

import { motion, AnimatePresence } from "framer-motion"
import {
    MagicWand,
    ChatText,
    Check,
    TextAa,
    ArrowsClockwise,
    Sparkle,
    CaretRight,
    Spinner
} from "@phosphor-icons/react"
import { useState } from "react"
import { cn } from "@/lib/utils"

interface FloatingAIMenuProps {
    position: { x: number; y: number }
    isVisible: boolean
    onAction: (action: string, subAction?: string) => void
    isProcessing: boolean
}

const MENU_ITEMS = [
    {
        id: "comment",
        label: "Add Comment",
        icon: ChatText,
        color: "text-violet-400",
        hoverBg: "hover:bg-violet-500/20",
    },
    {
        id: "rewrite",
        label: "Rewrite",
        icon: ArrowsClockwise,
        color: "text-blue-400",
        hoverBg: "hover:bg-blue-500/20",
        hasSubmenu: true,
        submenu: [
            { id: "professional", label: "Professional" },
            { id: "casual", label: "Casual" },
            { id: "concise", label: "Concise" },
            { id: "user-style", label: "Your Style" },
        ]
    },
    {
        id: "grammar",
        label: "Fix Grammar",
        icon: Check,
        color: "text-green-400",
        hoverBg: "hover:bg-green-500/20",
    },
    {
        id: "improve",
        label: "Improve",
        icon: Sparkle,
        color: "text-amber-400",
        hoverBg: "hover:bg-amber-500/20",
    },
]

export function FloatingAIMenu({ position, isVisible, onAction, isProcessing }: FloatingAIMenuProps) {
    const [activeSubmenu, setActiveSubmenu] = useState<string | null>(null)

    const handleItemClick = (item: typeof MENU_ITEMS[0]) => {
        if (item.hasSubmenu) {
            setActiveSubmenu(activeSubmenu === item.id ? null : item.id)
        } else {
            onAction(item.id)
            setActiveSubmenu(null)
        }
    }

    const handleSubmenuClick = (parentId: string, subId: string) => {
        onAction(parentId, subId)
        setActiveSubmenu(null)
    }

    return (
        <AnimatePresence>
            {isVisible && (
                <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.95 }}
                    transition={{
                        type: "spring",
                        bounce: 0.3,
                        duration: 0.4
                    }}
                    style={{
                        position: 'fixed',
                        left: position.x,
                        top: position.y,
                        transform: "translateX(-50%)"
                    }}
                    className="z-[100] origin-bottom"
                    onMouseLeave={() => setActiveSubmenu(null)}
                >
                    {/* Main Menu */}
                    <div className="flex items-center gap-0.5 p-1.5 rounded-2xl bg-black/90 backdrop-blur-xl border border-white/10 shadow-2xl shadow-black/50">
                        {isProcessing ? (
                            <div className="flex items-center gap-2.5 px-4 py-2">
                                <Spinner size={16} className="animate-spin text-violet-400" />
                                <span className="text-xs font-medium text-white/80">Processing...</span>
                            </div>
                        ) : (
                            MENU_ITEMS.map((item) => (
                                <div key={item.id} className="relative">
                                    <motion.button
                                        whileHover={{ scale: 1.05 }}
                                        whileTap={{ scale: 0.95 }}
                                        onClick={() => handleItemClick(item)}
                                        className={cn(
                                            "flex items-center gap-2 px-3 py-2 rounded-xl transition-colors text-xs font-medium text-white/80 hover:text-white",
                                            item.hoverBg,
                                            activeSubmenu === item.id && "bg-white/10"
                                        )}
                                    >
                                        <item.icon size={14} className={item.color} weight="fill" />
                                        <span>{item.label}</span>
                                        {item.hasSubmenu && (
                                            <CaretRight
                                                size={10}
                                                className={cn(
                                                    "text-white/40 transition-transform",
                                                    activeSubmenu === item.id && "rotate-90"
                                                )}
                                            />
                                        )}
                                    </motion.button>

                                    {/* Submenu */}
                                    <AnimatePresence>
                                        {item.hasSubmenu && activeSubmenu === item.id && (
                                            <motion.div
                                                initial={{ opacity: 0, y: -8, scale: 0.95 }}
                                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                                exit={{ opacity: 0, y: -8, scale: 0.95 }}
                                                transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                                                className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 py-1.5 px-1 rounded-xl bg-black/95 backdrop-blur-xl border border-white/10 shadow-xl min-w-[130px]"
                                            >
                                                {item.submenu?.map((sub) => (
                                                    <motion.button
                                                        key={sub.id}
                                                        whileHover={{ x: 2 }}
                                                        onClick={() => handleSubmenuClick(item.id, sub.id)}
                                                        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-white/70 hover:text-white hover:bg-white/10 transition-colors"
                                                    >
                                                        {sub.label}
                                                    </motion.button>
                                                ))}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            ))
                        )}
                    </div>

                    {/* Subtle glow effect */}
                    <div className="absolute inset-0 -z-10 rounded-2xl bg-gradient-to-b from-violet-500/20 to-blue-500/20 blur-xl opacity-50" />
                </motion.div>
            )}
        </AnimatePresence>
    )
}
