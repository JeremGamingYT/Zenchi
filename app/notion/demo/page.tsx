"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import type { Editor } from '@tiptap/react'
import {
    Image as ImageIcon,
    DotsThreeVertical,
    ClockCounterClockwise,
    ShareNetwork,
    Star,
    ChatText
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { NotionLayout } from "@/app/components/notion/notion-layout"
import { TiptapEditor } from "@/app/components/notion/tiptap-editor"
import { FloatingAIMenu } from "@/app/components/notion/floating-ai-menu"
import { AISuggestionPopup } from "@/app/components/notion/ai-suggestion-popup"
import { CommentBubble } from "@/app/components/notion/comment-bubble"
import { FileDropZone } from "@/app/components/notion/file-drop-zone"
import { EditorToolbar } from "@/app/components/notion/editor-toolbar"
import { SlashCommandMenu } from "@/app/components/notion/slash-command-menu"

// Mock AI responses for demo
const MOCK_AI_RESPONSES: Record<string, Record<string, string>> = {
    grammar: {
        default: "Grammar has been corrected with proper punctuation and sentence structure."
    },
    improve: {
        default: "This sentence has been enhanced for clarity and impact while maintaining your original meaning."
    },
    rewrite: {
        professional: "This has been rewritten in a formal, professional tone suitable for business communications.",
        casual: "Here's a more casual, friendly version of your text!",
        concise: "Shortened for clarity.",
        "user-style": "Rewritten to match your personal writing style based on previous content."
    }
}

export default function EditorPage() {
    const editorRef = useRef<Editor | null>(null)

    // Selection state
    const [floatingMenuPos, setFloatingMenuPos] = useState({ x: 0, y: 0 })
    const [showMenu, setShowMenu] = useState(false)
    const [selectedText, setSelectedText] = useState("")
    const [selectionRange, setSelectionRange] = useState({ from: 0, to: 0 })

    // AI Processing
    const [isAiProcessing, setIsAiProcessing] = useState(false)

    // Suggestion popup
    const [suggestionPopup, setSuggestionPopup] = useState<{
        visible: boolean
        position: { x: number; y: number }
        originalText: string
        suggestedText: string
        type: 'grammar' | 'style' | 'improve'
    }>({
        visible: false,
        position: { x: 0, y: 0 },
        originalText: "",
        suggestedText: "",
        type: 'improve'
    })

    // Comment system
    const [commentBubble, setCommentBubble] = useState<{
        visible: boolean
        position: { x: number; y: number }
        comment: string
        markId: string
    }>({
        visible: false,
        position: { x: 0, y: 0 },
        comment: "",
        markId: ""
    })

    const [commentInput, setCommentInput] = useState<{
        visible: boolean
        position: { x: number; y: number }
    }>({
        visible: false,
        position: { x: 0, y: 0 }
    })
    const [pendingComment, setPendingComment] = useState("")

    // Slash command menu state
    const [slashCommand, setSlashCommand] = useState<{
        visible: boolean
        position: { x: number; y: number }
        query: string
    }>({
        visible: false,
        position: { x: 0, y: 0 },
        query: ""
    })

    // Handle slash command
    const handleSlashCommand = useCallback((show: boolean, position: { x: number; y: number }, query: string) => {
        setSlashCommand({ visible: show, position, query })
    }, [])

    const handleSlashClose = useCallback(() => {
        setSlashCommand(prev => ({ ...prev, visible: false }))
        // Remove the slash from the editor
        if (editorRef.current) {
            const { from } = editorRef.current.state.selection
            const textBefore = editorRef.current.state.doc.textBetween(Math.max(0, from - 20), from, '\n')
            const slashIndex = textBefore.lastIndexOf('/')
            if (slashIndex !== -1) {
                const deleteFrom = from - (textBefore.length - slashIndex)
                editorRef.current.chain().focus().deleteRange({ from: deleteFrom, to: from }).run()
            }
        }
    }, [])

    // Editor content
    const [editorContent, setEditorContent] = useState(`
        <h1>Welcome to Ziro Pages ✨</h1>
        <p>This is your AI-powered writing space. Here's what you can do:</p>
        <h2>AI Features</h2>
        <p>Select any text to see the AI menu appear. You can ask the AI to improve your writing, fix grammar issues, or add comments for later revision.</p>
        <p>Try selecting this paragraph and clicking "Improve" to see how the AI can enhance your writing!</p>
        <h2>Task Lists</h2>
        <ul data-type="taskList">
            <li data-type="taskItem" data-checked="true">Create your first document</li>
            <li data-type="taskItem" data-checked="false">Try the AI writing assistant</li>
            <li data-type="taskItem" data-checked="false">Upload a PDF or image</li>
            <li data-type="taskItem" data-checked="false">Share with your team</li>
        </ul>
        <h2>Getting Started</h2>
        <p>Start writing below and let the AI help you create amazing content. The more you write, the better the AI understands your style!</p>
    `)

    // Handle text selection
    const handleSelectionUpdate = useCallback((editor: Editor, text: string, from: number, to: number) => {
        if (text.length > 0 && !isAiProcessing) {
            const { view } = editor
            const coords = view.coordsAtPos(from)

            setFloatingMenuPos({
                x: coords.left + 100,
                y: coords.top - 60
            })
            setSelectedText(text)
            setSelectionRange({ from, to })
            setShowMenu(true)
        } else if (text.length === 0) {
            setShowMenu(false)
        }
    }, [isAiProcessing])

    // Handle AI actions
    const handleAiAction = useCallback(async (action: string, subAction?: string) => {
        if (!editorRef.current || !selectedText) return

        setIsAiProcessing(true)
        setShowMenu(false)

        // Simulate AI processing
        await new Promise(resolve => setTimeout(resolve, 1200))

        const editor = editorRef.current

        if (action === "comment") {
            // Show comment input
            setCommentInput({
                visible: true,
                position: floatingMenuPos
            })
            setIsAiProcessing(false)
            return
        }

        if (action === "grammar") {
            // Mock grammar fix
            const fixedText = selectedText.charAt(0).toUpperCase() + selectedText.slice(1)
            setSuggestionPopup({
                visible: true,
                position: { x: floatingMenuPos.x, y: floatingMenuPos.y + 20 },
                originalText: selectedText,
                suggestedText: fixedText.endsWith('.') ? fixedText : fixedText + '.',
                type: 'grammar'
            })
        } else if (action === "improve") {
            // Mock improvement
            const improved = `Enhanced: ${selectedText}`
            setSuggestionPopup({
                visible: true,
                position: { x: floatingMenuPos.x, y: floatingMenuPos.y + 20 },
                originalText: selectedText,
                suggestedText: improved,
                type: 'improve'
            })
        } else if (action === "rewrite") {
            // Mock rewrite based on style
            const style = subAction || 'professional'
            const rewritten = `[${style}] ${selectedText}`
            setSuggestionPopup({
                visible: true,
                position: { x: floatingMenuPos.x, y: floatingMenuPos.y + 20 },
                originalText: selectedText,
                suggestedText: rewritten,
                type: 'style'
            })
        }

        setIsAiProcessing(false)
    }, [selectedText, floatingMenuPos])

    // Accept suggestion
    const handleAcceptSuggestion = useCallback(() => {
        if (!editorRef.current) return

        const editor = editorRef.current
        editor
            .chain()
            .focus()
            .setTextSelection(selectionRange)
            .deleteSelection()
            .insertContent(suggestionPopup.suggestedText)
            .run()

        setSuggestionPopup(prev => ({ ...prev, visible: false }))
    }, [selectionRange, suggestionPopup.suggestedText])

    // Reject suggestion
    const handleRejectSuggestion = useCallback(() => {
        setSuggestionPopup(prev => ({ ...prev, visible: false }))
    }, [])

    // Submit comment
    const handleSubmitComment = useCallback(() => {
        if (!editorRef.current || !pendingComment.trim()) return

        const editor = editorRef.current
        editor
            .chain()
            .focus()
            .setTextSelection(selectionRange)
            .setAIComment(pendingComment)
            .run()

        setPendingComment("")
        setCommentInput(prev => ({ ...prev, visible: false }))
    }, [pendingComment, selectionRange])

    // Handle clicking on marked comments
    useEffect(() => {
        const handleClick = (e: MouseEvent) => {
            const target = e.target as HTMLElement
            if (target.classList.contains('ai-comment-mark')) {
                const comment = target.getAttribute('data-comment')
                const markId = target.getAttribute('data-comment-id')
                if (comment && markId) {
                    const rect = target.getBoundingClientRect()
                    setCommentBubble({
                        visible: true,
                        position: { x: rect.left + rect.width / 2, y: rect.bottom + 10 },
                        comment,
                        markId
                    })
                }
            }
        }

        document.addEventListener('click', handleClick)
        return () => document.removeEventListener('click', handleClick)
    }, [])

    // Delete comment
    const handleDeleteComment = useCallback(() => {
        if (!editorRef.current) return

        // For simplicity, just close the bubble
        // In production, you'd remove the mark from the editor
        setCommentBubble(prev => ({ ...prev, visible: false }))
    }, [])

    return (
        <NotionLayout>
            <div className="min-h-full flex flex-col font-sans">

                {/* Editor Top Bar */}
                <div className="h-14 flex items-center justify-between px-4 border-b border-border/20 bg-background/50 backdrop-blur-sm sticky top-0 z-40 shrink-0">
                    <div className="flex items-center text-sm text-muted-foreground gap-1">
                        <span className="hover:bg-muted/50 px-2 py-1 rounded cursor-pointer transition-colors">Private</span>
                        <span>/</span>
                        <span className="hover:bg-muted/50 px-2 py-1 rounded cursor-pointer transition-colors font-medium text-foreground">Welcome to Ziro Pages</span>
                    </div>

                    <div className="flex items-center gap-1">
                        <span className="text-xs text-muted-foreground mr-4 flex items-center gap-1">
                            <motion.div
                                className="size-2 rounded-full bg-green-500"
                                animate={{ opacity: [1, 0.5, 1] }}
                                transition={{ duration: 2, repeat: Infinity }}
                            />
                            Saved
                        </span>
                        <Button variant="ghost" size="sm" className="gap-2 h-8 text-muted-foreground hover:text-foreground">
                            <ClockCounterClockwise size={16} />
                        </Button>
                        <Button variant="ghost" size="sm" className="gap-2 h-8 text-muted-foreground hover:text-foreground">
                            <Star size={16} />
                        </Button>
                        <Button variant="outline" size="sm" className="gap-2 h-8 ml-2 bg-blue-600/10 text-blue-500 border-blue-500/20 hover:bg-blue-600/20 hover:text-blue-400">
                            <ShareNetwork size={16} /> Share
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 ml-1"><DotsThreeVertical size={16} /></Button>
                    </div>
                </div>

                {/* Editor Toolbar */}
                <EditorToolbar editor={editorRef.current} />

                {/* Editor Canvas Container */}
                <div className="flex-1 max-w-4xl mx-auto w-full px-12 py-8 relative">

                    {/* Cover Image Feature */}
                    <motion.div
                        whileHover={{ scale: 1.005 }}
                        transition={{ type: "spring", bounce: 0.2 }}
                        className="group relative w-full h-48 bg-gradient-to-br from-blue-500/10 via-violet-500/10 to-purple-500/10 rounded-2xl mb-8 flex items-center justify-center border border-border/30 hover:border-blue-500/30 transition-colors cursor-pointer overflow-hidden"
                    >
                        <div className="flex flex-col items-center gap-2 text-muted-foreground opacity-50 group-hover:opacity-100 transition-opacity">
                            <ImageIcon size={32} />
                            <span className="text-sm font-medium">Add Cover Image</span>
                        </div>
                        {/* Animated gradient overlay */}
                        <div className="absolute inset-0 opacity-30 bg-[radial-gradient(circle_at_50%_50%,rgba(139,92,246,0.1),transparent_70%)]" />
                        {/* Grid pattern */}
                        <div className="absolute inset-0 opacity-20 bg-[radial-gradient(#808080_1px,transparent_1px)] [background-size:20px_20px]" />
                    </motion.div>

                    {/* Icon + Title Input */}
                    <div className="group flex items-start -ml-1 mb-8">
                        <motion.div
                            whileHover={{ scale: 1.1, rotate: 5 }}
                            whileTap={{ scale: 0.95 }}
                            className="py-2 pr-4 pl-1 opacity-50 group-hover:opacity-100 transition-opacity cursor-pointer"
                        >
                            <div className="size-10 rounded-lg flex items-center justify-center hover:bg-muted/50 font-emoji text-3xl transition-colors">
                                ✨
                            </div>
                        </motion.div>
                        <input
                            type="text"
                            placeholder="Untitled"
                            className="w-full bg-transparent text-5xl font-bold placeholder:text-muted-foreground/20 focus:outline-none tracking-tight"
                            defaultValue="Welcome to Ziro Pages"
                        />
                    </div>

                    {/* File Attachments Section */}
                    <div className="mb-10">
                        <FileDropZone onFilesAdded={(files) => console.log('Files added:', files)} />
                    </div>

                    {/* Real Rich Text Editor */}
                    <TiptapEditor
                        content={editorContent}
                        onChange={setEditorContent}
                        onSelectionUpdate={handleSelectionUpdate}
                        editorRef={editorRef}
                        onSlashCommand={handleSlashCommand}
                    />

                </div>

                {/* Slash Command Menu */}
                <SlashCommandMenu
                    editor={editorRef.current}
                    isVisible={slashCommand.visible}
                    position={slashCommand.position}
                    query={slashCommand.query}
                    onClose={handleSlashClose}
                />

                {/* Floating AI Menu */}
                <FloatingAIMenu
                    position={floatingMenuPos}
                    isVisible={showMenu}
                    onAction={handleAiAction}
                    isProcessing={isAiProcessing}
                />

                {/* AI Suggestion Popup */}
                <AISuggestionPopup
                    isVisible={suggestionPopup.visible}
                    position={suggestionPopup.position}
                    originalText={suggestionPopup.originalText}
                    suggestedText={suggestionPopup.suggestedText}
                    type={suggestionPopup.type}
                    onAccept={handleAcceptSuggestion}
                    onReject={handleRejectSuggestion}
                />

                {/* Comment Bubble */}
                <CommentBubble
                    isVisible={commentBubble.visible}
                    position={commentBubble.position}
                    comment={commentBubble.comment}
                    onDelete={handleDeleteComment}
                    onClose={() => setCommentBubble(prev => ({ ...prev, visible: false }))}
                />

                {/* Comment Input Modal */}
                <AnimatePresence>
                    {commentInput.visible && (
                        <motion.div
                            initial={{ opacity: 0, y: 8, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 8, scale: 0.95 }}
                            transition={{ type: "spring", bounce: 0.25 }}
                            style={{
                                position: 'fixed',
                                left: commentInput.position.x,
                                top: commentInput.position.y,
                                transform: "translateX(-50%)"
                            }}
                            className="z-[110] w-80"
                        >
                            <div className="rounded-xl bg-violet-950/95 backdrop-blur-xl border border-violet-500/20 shadow-2xl overflow-hidden">
                                <div className="px-4 py-3 border-b border-violet-500/20 flex items-center gap-2 bg-violet-500/10">
                                    <ChatText size={16} className="text-violet-400" weight="fill" />
                                    <span className="text-sm font-semibold text-white">Add Comment for AI</span>
                                </div>
                                <div className="p-4">
                                    <textarea
                                        value={pendingComment}
                                        onChange={(e) => setPendingComment(e.target.value)}
                                        placeholder="What should the AI do with this text later?"
                                        className="w-full h-24 bg-black/30 border border-violet-500/20 rounded-lg px-3 py-2 text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-violet-500/50 resize-none"
                                        autoFocus
                                    />
                                    <div className="flex gap-2 mt-3">
                                        <Button
                                            onClick={handleSubmitComment}
                                            className="flex-1 bg-violet-600 hover:bg-violet-500 text-white border-0"
                                            size="sm"
                                        >
                                            Add Comment
                                        </Button>
                                        <Button
                                            onClick={() => {
                                                setCommentInput(prev => ({ ...prev, visible: false }))
                                                setPendingComment("")
                                            }}
                                            variant="ghost"
                                            size="sm"
                                            className="text-white/60 hover:text-white hover:bg-white/10"
                                        >
                                            Cancel
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>
        </NotionLayout>
    )
}
