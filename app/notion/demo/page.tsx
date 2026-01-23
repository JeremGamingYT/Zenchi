"use client"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
    TextAa,
    MagicWand,
    ChatText,
    Check,
    FilePdf,
    Image as ImageIcon,
    DotsThreeVertical,
    ClockCounterClockwise,
    ShareNetwork,
    Star
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { NotionLayout } from "@/app/components/notion/notion-layout"

// --- Fake AI Logic ---
const MOCK_SUGGESTIONS = [
    { id: 1, text: "The quick brown fox", original: "The fast brown fox", type: "rewrite" },
]

export default function EditorPage() {
    const contentRef = useRef<HTMLDivElement>(null)
    const [selectionRange, setSelectionRange] = useState<Range | null>(null)
    const [floatingMenuPos, setFloatingMenuPos] = useState({ x: 0, y: 0 })
    const [showMenu, setShowMenu] = useState(false)
    const [isAiProcessing, setIsAiProcessing] = useState(false)

    // Handle Text Selection
    useEffect(() => {
        const handleSelection = () => {
            const selection = window.getSelection()
            if (selection && selection.rangeCount > 0 && !selection.isCollapsed) {
                const range = selection.getRangeAt(0)
                // Check if selection is inside editor
                if (contentRef.current?.contains(range.commonAncestorContainer)) {
                    const rect = range.getBoundingClientRect()
                    setSelectionRange(range)
                    setFloatingMenuPos({
                        x: rect.left + rect.width / 2,
                        y: rect.top - 10
                    })
                    setShowMenu(true)
                    return
                }
            }
            setShowMenu(false)
        }

        document.addEventListener("selectionchange", handleSelection)
        return () => document.removeEventListener("selectionchange", handleSelection)
    }, [])

    const handleAiAction = (action: string) => {
        setIsAiProcessing(true)
        setTimeout(() => {
            setIsAiProcessing(false)
            // Mock action
            console.log("AI Action:", action)
            setShowMenu(false)
        }, 1500)
    }

    return (
        <NotionLayout>
            <div className="min-h-full flex flex-col font-sans">

                {/* Editor Top Bar (Specific to Page) */}
                <div className="h-12 flex items-center justify-between px-4 mt-2 mb-4 shrink-0">
                    <div className="flex items-center text-sm text-muted-foreground gap-1">
                        <span className="hover:bg-muted/50 px-2 py-1 rounded cursor-pointer transition-colors">Private</span>
                        <span>/</span>
                        <span className="hover:bg-muted/50 px-2 py-1 rounded cursor-pointer transition-colors font-medium text-foreground">Product Requirements</span>
                    </div>

                    <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground mr-4">Saved</span>
                        <Button variant="ghost" size="sm" className="gap-2 h-8 text-muted-foreground">
                            <ClockCounterClockwise size={16} />
                        </Button>
                        <Button variant="ghost" size="sm" className="gap-2 h-8 text-muted-foreground">
                            <Star size={16} />
                        </Button>
                        <Button variant="outline" size="sm" className="gap-2 h-8 ml-2 bg-blue-600/10 text-blue-500 border-blue-500/20 hover:bg-blue-600/20">
                            <ShareNetwork size={16} /> Share
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 ml-1"><DotsThreeVertical size={16} /></Button>
                    </div>
                </div>

                {/* Editor Canvas */}
                <div className="flex-1 max-w-4xl mx-auto w-full px-12 pb-24 relative">

                    {/* Cover Image */}
                    <div className="group relative w-full h-48 bg-gradient-to-r from-blue-500/5 to-violet-500/5 rounded-xl mb-8 flex items-center justify-center border border-dashed border-border/40 hover:bg-muted/30 transition-colors cursor-pointer overflow-hidden">
                        <div className="flex flex-col items-center gap-2 text-muted-foreground opacity-50 group-hover:opacity-100 transition-opacity">
                            <ImageIcon size={32} />
                            <span className="text-sm font-medium">Add Cover Image</span>
                        </div>
                        {/* Subtle pattern overlay */}
                        <div className="absolute inset-0 opacity-20 bg-[radial-gradient(#808080_1px,transparent_1px)] [background-size:16px_16px]" />
                    </div>

                    {/* Icon + Title */}
                    <div className="group flex items-start -ml-1 mb-8">
                        <div className="py-2 pr-4 pl-1 opacity-50 group-hover:opacity-100 transition-opacity cursor-pointer">
                            <div className="size-8 rounded flex items-center justify-center hover:bg-muted font-emoji text-2xl">
                                📄
                            </div>
                        </div>
                        <input
                            type="text"
                            placeholder="Untitled"
                            className="w-full bg-transparent text-5xl font-bold placeholder:text-muted-foreground/20 focus:outline-none"
                            defaultValue="Product Requirements"
                        />
                    </div>

                    {/* Content Area */}
                    <div
                        ref={contentRef}
                        contentEditable
                        className="prose prose-lg dark:prose-invert max-w-none outline-none min-h-[50vh] empty:before:content-['Start_writing_with_AI...'] empty:before:text-muted-foreground/30 selection:bg-blue-500/30"
                        suppressContentEditableWarning
                    >
                        <p>Zenchi Pages is designed to be a living document system.</p>
                        <p>Try selecting this text to see the AI menu appear. It feels fluid, responsive, and helper-focused.</p>
                        <p>We can also add PDFs directly here:</p>

                        <div className="not-prose my-6 p-4 border border-border bg-card/40 rounded-lg flex items-center gap-4 group cursor-pointer hover:border-blue-500/50 hover:shadow-md transition-all">
                            <div className="size-10 rounded bg-red-500/10 text-red-500 flex items-center justify-center">
                                <FilePdf size={24} weight="fill" />
                            </div>
                            <div className="flex-1">
                                <div className="font-medium group-hover:text-blue-500 transition-colors">Spec_v1.pdf</div>
                                <div className="text-xs text-muted-foreground">2.4 MB • Uploaded just now</div>
                            </div>
                            <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity bg-background/50">Details</Button>
                        </div>

                        <h3>Next Steps</h3>
                        <ul className="todo-list">
                            <li className="flex gap-2 items-center"><div className="size-4 border rounded border-foreground/50"></div> Define core features</li>
                            <li className="flex gap-2 items-center"><div className="size-4 border rounded border-foreground/50"></div> Design database schema</li>
                        </ul>

                        <p>Continue writing your masterpiece here...</p>
                    </div>
                </div>

                {/* Floating AI Menu */}
                <AnimatePresence>
                    {showMenu && (
                        <motion.div
                            initial={{ opacity: 0, y: 10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                            style={{
                                position: 'fixed',
                                left: floatingMenuPos.x,
                                top: floatingMenuPos.y - 50,
                                transform: "translateX(-50%)"
                            }}
                            className="z-50 flex items-center gap-1 p-1 rounded-full bg-black/80 text-white backdrop-blur-xl border border-white/10 shadow-2xl origin-bottom"
                        >
                            {isAiProcessing ? (
                                <div className="flex items-center gap-2 px-3 py-1.5 min-w-[120px]">
                                    <MagicWand size={16} className="animate-spin text-purple-400" />
                                    <span className="text-xs font-medium">Thinking...</span>
                                </div>
                            ) : (
                                <>
                                    <button
                                        onClick={() => handleAiAction("improve")}
                                        className="flex items-center gap-2 px-3 py-1.5 hover:bg-white/20 rounded-full transition-colors text-xs font-medium"
                                    >
                                        <MagicWand size={14} className="text-purple-400" />
                                        Improve
                                    </button>
                                    <div className="w-px h-4 bg-white/20" />
                                    <button
                                        onClick={() => handleAiAction("grammar")}
                                        className="flex items-center gap-2 px-3 py-1.5 hover:bg-white/20 rounded-full transition-colors text-xs font-medium"
                                    >
                                        <Check size={14} className="text-green-400" />
                                        Fix Grammar
                                    </button>
                                    <div className="w-px h-4 bg-white/20" />
                                    <button
                                        onClick={() => handleAiAction("comment")}
                                        className="p-1.5 hover:bg-white/20 rounded-full transition-colors"
                                    >
                                        <ChatText size={14} />
                                    </button>
                                </>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>
        </NotionLayout>
    )
}
