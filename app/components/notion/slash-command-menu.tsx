"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Editor, Range } from '@tiptap/core'
import {
    TextHOne,
    TextHTwo,
    TextHThree,
    ListBullets,
    ListNumbers,
    ListChecks,
    Quotes,
    Code,
    CodeBlock,
    Image,
    Table,
    Minus,
    MagicWand,
    FileText,
    Video,
    LinkSimple,
    Columns,
    Calendar,
    Warning,
    Info,
    LightbulbFilament
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"

interface SlashCommandMenuProps {
    editor: Editor | null
    isVisible: boolean
    position: { x: number; y: number }
    onClose: () => void
    query: string
}

interface CommandItem {
    id: string
    title: string
    description: string
    icon: React.ElementType
    category: string
    action: (editor: Editor) => void
}

const COMMANDS: CommandItem[] = [
    // Text
    {
        id: 'heading1',
        title: 'Heading 1',
        description: 'Large section heading',
        icon: TextHOne,
        category: 'Text',
        action: (editor) => editor.chain().focus().toggleHeading({ level: 1 }).run()
    },
    {
        id: 'heading2',
        title: 'Heading 2',
        description: 'Medium section heading',
        icon: TextHTwo,
        category: 'Text',
        action: (editor) => editor.chain().focus().toggleHeading({ level: 2 }).run()
    },
    {
        id: 'heading3',
        title: 'Heading 3',
        description: 'Small section heading',
        icon: TextHThree,
        category: 'Text',
        action: (editor) => editor.chain().focus().toggleHeading({ level: 3 }).run()
    },
    // Lists
    {
        id: 'bulletList',
        title: 'Bullet List',
        description: 'Create a simple bullet list',
        icon: ListBullets,
        category: 'Lists',
        action: (editor) => editor.chain().focus().toggleBulletList().run()
    },
    {
        id: 'numberedList',
        title: 'Numbered List',
        description: 'Create a numbered list',
        icon: ListNumbers,
        category: 'Lists',
        action: (editor) => editor.chain().focus().toggleOrderedList().run()
    },
    {
        id: 'todoList',
        title: 'To-do List',
        description: 'Track tasks with checkboxes',
        icon: ListChecks,
        category: 'Lists',
        action: (editor) => editor.chain().focus().toggleTaskList().run()
    },
    // Blocks
    {
        id: 'quote',
        title: 'Quote',
        description: 'Capture a quote',
        icon: Quotes,
        category: 'Blocks',
        action: (editor) => editor.chain().focus().toggleBlockquote().run()
    },
    {
        id: 'codeBlock',
        title: 'Code Block',
        description: 'Capture a code snippet',
        icon: CodeBlock,
        category: 'Blocks',
        action: (editor) => editor.chain().focus().toggleCodeBlock().run()
    },
    {
        id: 'divider',
        title: 'Divider',
        description: 'Horizontal dividing line',
        icon: Minus,
        category: 'Blocks',
        action: (editor) => editor.chain().focus().setHorizontalRule().run()
    },
    // Media
    {
        id: 'image',
        title: 'Image',
        description: 'Upload or embed an image',
        icon: Image,
        category: 'Media',
        action: (editor) => {
            const url = window.prompt('Enter image URL:')
            if (url) editor.chain().focus().setImage({ src: url }).run()
        }
    },
    {
        id: 'table',
        title: 'Table',
        description: 'Add a table',
        icon: Table,
        category: 'Media',
        action: (editor) => editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()
    },
    // Callouts
    {
        id: 'calloutInfo',
        title: 'Info Callout',
        description: 'Highlight important information',
        icon: Info,
        category: 'Callouts',
        action: (editor) => {
            editor.chain().focus().insertContent({
                type: 'blockquote',
                content: [{ type: 'paragraph', content: [{ type: 'text', text: 'ℹ️ Info: Your message here' }] }]
            }).run()
        }
    },
    {
        id: 'calloutTip',
        title: 'Tip Callout',
        description: 'Share a helpful tip',
        icon: LightbulbFilament,
        category: 'Callouts',
        action: (editor) => {
            editor.chain().focus().insertContent({
                type: 'blockquote',
                content: [{ type: 'paragraph', content: [{ type: 'text', text: '💡 Tip: Your tip here' }] }]
            }).run()
        }
    },
    {
        id: 'calloutWarning',
        title: 'Warning Callout',
        description: 'Highlight a warning',
        icon: Warning,
        category: 'Callouts',
        action: (editor) => {
            editor.chain().focus().insertContent({
                type: 'blockquote',
                content: [{ type: 'paragraph', content: [{ type: 'text', text: '⚠️ Warning: Your warning here' }] }]
            }).run()
        }
    },
    // AI
    {
        id: 'aiWrite',
        title: 'AI Write',
        description: 'Let AI help you write',
        icon: MagicWand,
        category: 'AI',
        action: (editor) => {
            editor.chain().focus().insertContent('✨ AI is thinking...').run()
        }
    },
]

export function SlashCommandMenu({ editor, isVisible, position, onClose, query }: SlashCommandMenuProps) {
    const [selectedIndex, setSelectedIndex] = useState(0)
    const menuRef = useRef<HTMLDivElement>(null)

    // Filter commands based on query
    const filteredCommands = COMMANDS.filter(cmd =>
        cmd.title.toLowerCase().includes(query.toLowerCase()) ||
        cmd.description.toLowerCase().includes(query.toLowerCase())
    )

    // Group by category
    const groupedCommands = filteredCommands.reduce((acc, cmd) => {
        if (!acc[cmd.category]) acc[cmd.category] = []
        acc[cmd.category].push(cmd)
        return acc
    }, {} as Record<string, CommandItem[]>)

    // Reset selection when query changes
    useEffect(() => {
        setSelectedIndex(0)
    }, [query])

    // Handle keyboard navigation
    useEffect(() => {
        if (!isVisible) return

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSelectedIndex(i => Math.min(i + 1, filteredCommands.length - 1))
            } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSelectedIndex(i => Math.max(i - 1, 0))
            } else if (e.key === 'Enter') {
                e.preventDefault()
                if (filteredCommands[selectedIndex] && editor) {
                    filteredCommands[selectedIndex].action(editor)
                    onClose()
                }
            } else if (e.key === 'Escape') {
                onClose()
            }
        }

        document.addEventListener('keydown', handleKeyDown)
        return () => document.removeEventListener('keydown', handleKeyDown)
    }, [isVisible, filteredCommands, selectedIndex, editor, onClose])

    // Handle click outside
    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
                onClose()
            }
        }

        if (isVisible) {
            document.addEventListener('mousedown', handleClickOutside)
        }
        return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [isVisible, onClose])

    const executeCommand = (cmd: CommandItem) => {
        if (editor) {
            cmd.action(editor)
            onClose()
        }
    }

    let flatIndex = 0

    return (
        <AnimatePresence>
            {isVisible && filteredCommands.length > 0 && (
                <motion.div
                    ref={menuRef}
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                    style={{
                        position: 'fixed',
                        left: position.x,
                        top: position.y,
                    }}
                    className="z-[100] w-72 max-h-80 overflow-y-auto rounded-xl bg-popover/95 backdrop-blur-xl border border-border shadow-2xl"
                >
                    {Object.entries(groupedCommands).map(([category, commands]) => (
                        <div key={category}>
                            <div className="px-3 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wider border-b border-border/50 bg-muted/30">
                                {category}
                            </div>
                            {commands.map((cmd) => {
                                const itemIndex = flatIndex++
                                const isSelected = itemIndex === selectedIndex

                                return (
                                    <motion.button
                                        key={cmd.id}
                                        onClick={() => executeCommand(cmd)}
                                        className={cn(
                                            "w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors",
                                            "hover:bg-accent",
                                            isSelected && "bg-accent"
                                        )}
                                    >
                                        <div className={cn(
                                            "size-9 rounded-lg flex items-center justify-center",
                                            "bg-muted text-muted-foreground",
                                            isSelected && "bg-primary/10 text-primary"
                                        )}>
                                            <cmd.icon size={20} weight="duotone" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="text-sm font-medium truncate">{cmd.title}</div>
                                            <div className="text-xs text-muted-foreground truncate">{cmd.description}</div>
                                        </div>
                                    </motion.button>
                                )
                            })}
                        </div>
                    ))}

                    {filteredCommands.length === 0 && (
                        <div className="p-4 text-center text-sm text-muted-foreground">
                            No commands found
                        </div>
                    )}
                </motion.div>
            )}
        </AnimatePresence>
    )
}
