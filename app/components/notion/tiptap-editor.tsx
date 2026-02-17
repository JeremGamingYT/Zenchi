"use client"

import { useEditor, EditorContent, Editor } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import Link from '@tiptap/extension-link'
import Image from '@tiptap/extension-image'
import TaskList from '@tiptap/extension-task-list'
import TaskItem from '@tiptap/extension-task-item'
import Placeholder from '@tiptap/extension-placeholder'
import Highlight from '@tiptap/extension-highlight'
import TextAlign from '@tiptap/extension-text-align'
import { TextStyle } from '@tiptap/extension-text-style'
import { Color } from '@tiptap/extension-color'
import Subscript from '@tiptap/extension-subscript'
import Superscript from '@tiptap/extension-superscript'
import Typography from '@tiptap/extension-typography'
import { Table } from '@tiptap/extension-table'
import { TableRow } from '@tiptap/extension-table-row'
import { TableCell } from '@tiptap/extension-table-cell'
import { TableHeader } from '@tiptap/extension-table-header'
import HorizontalRule from '@tiptap/extension-horizontal-rule'
import Dropcursor from '@tiptap/extension-dropcursor'
import Gapcursor from '@tiptap/extension-gapcursor'
import { AIComment } from './extensions/ai-comment-mark'
import { AISuggestion } from './extensions/ai-suggestion-mark'

interface TiptapEditorProps {
    content?: string
    onChange?: (content: string) => void
    onSelectionUpdate?: (editor: Editor, selectedText: string, from: number, to: number) => void
    editorRef?: React.MutableRefObject<Editor | null>
    onSlashCommand?: (show: boolean, position: { x: number; y: number }, query: string) => void
}

export function TiptapEditor({ content, onChange, onSelectionUpdate, editorRef, onSlashCommand }: TiptapEditorProps) {
    const editor = useEditor({
        immediatelyRender: false,
        extensions: [
            StarterKit.configure({
                heading: {
                    levels: [1, 2, 3],
                },
                codeBlock: {
                    HTMLAttributes: {
                        class: 'rounded-lg bg-muted p-4 font-mono text-sm overflow-x-auto',
                    },
                },
                blockquote: {
                    HTMLAttributes: {
                        class: 'border-l-4 border-primary/30 pl-4 italic text-muted-foreground',
                    },
                },
            }),
            Underline,
            Link.configure({
                openOnClick: false,
                HTMLAttributes: {
                    class: 'text-primary underline underline-offset-2 hover:text-primary/80 transition-colors',
                },
            }),
            Image.configure({
                HTMLAttributes: {
                    class: 'rounded-lg max-w-full h-auto my-4',
                },
            }),
            TaskList.configure({
                HTMLAttributes: {
                    class: 'not-prose pl-0',
                },
            }),
            TaskItem.configure({
                nested: true,
                HTMLAttributes: {
                    class: 'flex items-start gap-2 my-1',
                },
            }),
            Placeholder.configure({
                placeholder: ({ node }) => {
                    if (node.type.name === 'heading') {
                        return `Heading ${node.attrs.level}`
                    }
                    return "Type '/' for commands, or start writing..."
                },
            }),
            Highlight.configure({
                multicolor: true,
            }),
            TextAlign.configure({
                types: ['heading', 'paragraph'],
            }),
            TextStyle,
            Color,
            Subscript,
            Superscript,
            Typography,
            Table.configure({
                resizable: true,
                HTMLAttributes: {
                    class: 'border-collapse table-auto w-full my-4',
                },
            }),
            TableRow,
            TableCell.configure({
                HTMLAttributes: {
                    class: 'border border-border p-2 min-w-[100px]',
                },
            }),
            TableHeader.configure({
                HTMLAttributes: {
                    class: 'border border-border p-2 bg-muted font-semibold min-w-[100px]',
                },
            }),
            HorizontalRule.configure({
                HTMLAttributes: {
                    class: 'my-6 border-t border-border',
                },
            }),
            Dropcursor.configure({
                color: 'hsl(var(--primary))',
                width: 2,
            }),
            Gapcursor,
            AIComment,
            AISuggestion,
        ],
        content: content || '<p>Start writing with AI help...</p>',
        editorProps: {
            attributes: {
                class: 'prose prose-lg dark:prose-invert max-w-none focus:outline-none min-h-[60vh] px-1 prose-headings:font-bold prose-headings:tracking-tight prose-h1:text-4xl prose-h2:text-2xl prose-h3:text-xl prose-p:text-foreground/80 prose-p:leading-relaxed prose-strong:text-foreground prose-a:text-primary hover:prose-a:text-primary/80 prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:before:content-none prose-code:after:content-none',
            },
            handleKeyDown: (view, event) => {
                // Detect "/" key for slash commands
                if (event.key === '/' && onSlashCommand) {
                    const { from } = view.state.selection
                    const coords = view.coordsAtPos(from)

                    // Delay to allow the "/" to be typed first
                    setTimeout(() => {
                        onSlashCommand(true, { x: coords.left, y: coords.bottom + 8 }, '')
                    }, 10)
                }
                return false
            },
        },
        onUpdate: ({ editor }) => {
            onChange?.(editor.getHTML())

            // Check for slash command query
            if (onSlashCommand) {
                const { from } = editor.state.selection
                const textBefore = editor.state.doc.textBetween(Math.max(0, from - 20), from, '\n')
                const slashIndex = textBefore.lastIndexOf('/')

                if (slashIndex !== -1) {
                    const query = textBefore.slice(slashIndex + 1)
                    const coords = editor.view.coordsAtPos(from)
                    onSlashCommand(true, { x: coords.left, y: coords.bottom + 8 }, query)
                } else {
                    onSlashCommand(false, { x: 0, y: 0 }, '')
                }
            }
        },
        onSelectionUpdate: ({ editor }) => {
            const { from, to } = editor.state.selection
            const text = editor.state.doc.textBetween(from, to, '')
            onSelectionUpdate?.(editor, text, from, to)
        },
        onCreate: ({ editor }) => {
            if (editorRef) {
                editorRef.current = editor
            }
        },
    })

    // Update ref when editor changes
    if (editorRef && editor) {
        editorRef.current = editor
    }

    if (!editor) {
        return (
            <div className="min-h-[60vh] flex items-center justify-center">
                <div className="animate-pulse text-muted-foreground">Loading editor...</div>
            </div>
        )
    }

    return (
        <EditorContent editor={editor} />
    )
}
