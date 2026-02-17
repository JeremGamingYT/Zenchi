"use client"

import { Editor } from '@tiptap/react'
import { motion } from "framer-motion"
import {
    TextB,
    TextItalic,
    TextUnderline,
    TextStrikethrough,
    TextHOne,
    TextHTwo,
    TextHThree,
    ListBullets,
    ListNumbers,
    ListChecks,
    Quotes,
    Code,
    CodeBlock,
    Link,
    Image,
    Table,
    Minus,
    TextAlignLeft,
    TextAlignCenter,
    TextAlignRight,
    TextAlignJustify,
    HighlighterCircle,
    Palette,
    Eraser,
    TextSubscript,
    TextSuperscript,
    ArrowUUpLeft,
    ArrowUUpRight
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { useState } from "react"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    DropdownMenuSeparator,
    DropdownMenuSub,
    DropdownMenuSubTrigger,
    DropdownMenuSubContent,
} from "@/components/ui/dropdown-menu"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"

interface EditorToolbarProps {
    editor: Editor | null
}

const HIGHLIGHT_COLORS = [
    { name: "Yellow", value: "#fef08a" },
    { name: "Green", value: "#bbf7d0" },
    { name: "Blue", value: "#bfdbfe" },
    { name: "Purple", value: "#ddd6fe" },
    { name: "Pink", value: "#fbcfe8" },
    { name: "Orange", value: "#fed7aa" },
]

const TEXT_COLORS = [
    { name: "Default", value: "inherit" },
    { name: "Gray", value: "#6b7280" },
    { name: "Red", value: "#ef4444" },
    { name: "Orange", value: "#f97316" },
    { name: "Yellow", value: "#eab308" },
    { name: "Green", value: "#22c55e" },
    { name: "Blue", value: "#3b82f6" },
    { name: "Purple", value: "#a855f7" },
    { name: "Pink", value: "#ec4899" },
]

function ToolbarButton({
    onClick,
    isActive = false,
    disabled = false,
    children,
    tooltip
}: {
    onClick: () => void
    isActive?: boolean
    disabled?: boolean
    children: React.ReactNode
    tooltip?: string
}) {
    return (
        <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onClick}
            disabled={disabled}
            title={tooltip}
            className={cn(
                "p-2 rounded-lg transition-colors",
                "hover:bg-muted text-muted-foreground hover:text-foreground",
                isActive && "bg-primary/10 text-primary",
                disabled && "opacity-50 cursor-not-allowed"
            )}
        >
            {children}
        </motion.button>
    )
}

function ToolbarDivider() {
    return <div className="w-px h-6 bg-border mx-1" />
}

export function EditorToolbar({ editor }: EditorToolbarProps) {
    const [linkUrl, setLinkUrl] = useState("")

    if (!editor) return null

    const setLink = () => {
        const previousUrl = editor.getAttributes('link').href
        const url = window.prompt('Enter URL:', previousUrl || 'https://')

        if (url === null) return
        if (url === '') {
            editor.chain().focus().extendMarkRange('link').unsetLink().run()
            return
        }

        editor.chain().focus().extendMarkRange('link').setLink({ href: url }).run()
    }

    const addImage = () => {
        const url = window.prompt('Enter image URL:')
        if (url) {
            editor.chain().focus().setImage({ src: url }).run()
        }
    }

    const addTable = () => {
        editor.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()
    }

    return (
        <div className="flex items-center gap-0.5 p-2 border-b border-border/50 bg-background/80 backdrop-blur-sm overflow-x-auto">
            {/* Undo / Redo */}
            <ToolbarButton
                onClick={() => editor.chain().focus().undo().run()}
                disabled={!editor.can().undo()}
                tooltip="Undo"
            >
                <ArrowUUpLeft size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().redo().run()}
                disabled={!editor.can().redo()}
                tooltip="Redo"
            >
                <ArrowUUpRight size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Text Formatting */}
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleBold().run()}
                isActive={editor.isActive('bold')}
                tooltip="Bold (Ctrl+B)"
            >
                <TextB size={18} weight="bold" />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleItalic().run()}
                isActive={editor.isActive('italic')}
                tooltip="Italic (Ctrl+I)"
            >
                <TextItalic size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleUnderline().run()}
                isActive={editor.isActive('underline')}
                tooltip="Underline (Ctrl+U)"
            >
                <TextUnderline size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleStrike().run()}
                isActive={editor.isActive('strike')}
                tooltip="Strikethrough"
            >
                <TextStrikethrough size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleCode().run()}
                isActive={editor.isActive('code')}
                tooltip="Inline Code"
            >
                <Code size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Subscript / Superscript */}
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleSubscript().run()}
                isActive={editor.isActive('subscript')}
                tooltip="Subscript"
            >
                <TextSubscript size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleSuperscript().run()}
                isActive={editor.isActive('superscript')}
                tooltip="Superscript"
            >
                <TextSuperscript size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Headings Dropdown */}
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="sm" className="gap-1 h-8 px-2">
                        <TextHOne size={18} />
                        <span className="text-xs">Heading</span>
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-[150px]">
                    <DropdownMenuItem onClick={() => editor.chain().focus().setParagraph().run()}>
                        <span className="text-sm">Paragraph</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}>
                        <TextHOne size={20} className="mr-2" />
                        <span className="text-xl font-bold">Heading 1</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}>
                        <TextHTwo size={18} className="mr-2" />
                        <span className="text-lg font-bold">Heading 2</span>
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}>
                        <TextHThree size={16} className="mr-2" />
                        <span className="text-base font-bold">Heading 3</span>
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <ToolbarDivider />

            {/* Text Color */}
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8" title="Text Color">
                        <Palette size={18} />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-[120px]">
                    {TEXT_COLORS.map((color) => (
                        <DropdownMenuItem
                            key={color.name}
                            onClick={() => editor.chain().focus().setColor(color.value).run()}
                            className="gap-2"
                        >
                            <div
                                className="size-4 rounded border border-border"
                                style={{ backgroundColor: color.value === 'inherit' ? 'currentColor' : color.value }}
                            />
                            <span>{color.name}</span>
                        </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => editor.chain().focus().unsetColor().run()}>
                        <Eraser size={14} className="mr-2" />
                        Remove Color
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            {/* Highlight */}
            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-8 w-8" title="Highlight">
                        <HighlighterCircle size={18} />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="min-w-[120px]">
                    {HIGHLIGHT_COLORS.map((color) => (
                        <DropdownMenuItem
                            key={color.name}
                            onClick={() => editor.chain().focus().toggleHighlight({ color: color.value }).run()}
                            className="gap-2"
                        >
                            <div
                                className="size-4 rounded border border-border"
                                style={{ backgroundColor: color.value }}
                            />
                            <span>{color.name}</span>
                        </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => editor.chain().focus().unsetHighlight().run()}>
                        <Eraser size={14} className="mr-2" />
                        Remove Highlight
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>

            <ToolbarDivider />

            {/* Alignment */}
            <ToolbarButton
                onClick={() => editor.chain().focus().setTextAlign('left').run()}
                isActive={editor.isActive({ textAlign: 'left' })}
                tooltip="Align Left"
            >
                <TextAlignLeft size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().setTextAlign('center').run()}
                isActive={editor.isActive({ textAlign: 'center' })}
                tooltip="Align Center"
            >
                <TextAlignCenter size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().setTextAlign('right').run()}
                isActive={editor.isActive({ textAlign: 'right' })}
                tooltip="Align Right"
            >
                <TextAlignRight size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().setTextAlign('justify').run()}
                isActive={editor.isActive({ textAlign: 'justify' })}
                tooltip="Justify"
            >
                <TextAlignJustify size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Lists */}
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleBulletList().run()}
                isActive={editor.isActive('bulletList')}
                tooltip="Bullet List"
            >
                <ListBullets size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleOrderedList().run()}
                isActive={editor.isActive('orderedList')}
                tooltip="Numbered List"
            >
                <ListNumbers size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleTaskList().run()}
                isActive={editor.isActive('taskList')}
                tooltip="Task List"
            >
                <ListChecks size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Block Elements */}
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleBlockquote().run()}
                isActive={editor.isActive('blockquote')}
                tooltip="Quote Block"
            >
                <Quotes size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().toggleCodeBlock().run()}
                isActive={editor.isActive('codeBlock')}
                tooltip="Code Block"
            >
                <CodeBlock size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={() => editor.chain().focus().setHorizontalRule().run()}
                tooltip="Horizontal Divider"
            >
                <Minus size={18} />
            </ToolbarButton>

            <ToolbarDivider />

            {/* Insert Elements */}
            <ToolbarButton
                onClick={setLink}
                isActive={editor.isActive('link')}
                tooltip="Add Link"
            >
                <Link size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={addImage}
                tooltip="Add Image"
            >
                <Image size={18} />
            </ToolbarButton>
            <ToolbarButton
                onClick={addTable}
                tooltip="Insert Table"
            >
                <Table size={18} />
            </ToolbarButton>

            {/* Table Controls (only show when in table) */}
            {editor.isActive('table') && (
                <>
                    <ToolbarDivider />
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="sm" className="gap-1 h-8 px-2 text-xs">
                                Table
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="start">
                            <DropdownMenuItem onClick={() => editor.chain().focus().addColumnBefore().run()}>
                                Add Column Before
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => editor.chain().focus().addColumnAfter().run()}>
                                Add Column After
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => editor.chain().focus().deleteColumn().run()}>
                                Delete Column
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => editor.chain().focus().addRowBefore().run()}>
                                Add Row Before
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => editor.chain().focus().addRowAfter().run()}>
                                Add Row After
                            </DropdownMenuItem>
                            <DropdownMenuItem onClick={() => editor.chain().focus().deleteRow().run()}>
                                Delete Row
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem onClick={() => editor.chain().focus().deleteTable().run()}>
                                Delete Table
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                </>
            )}

            {/* Clear Formatting */}
            <ToolbarButton
                onClick={() => editor.chain().focus().clearNodes().unsetAllMarks().run()}
                tooltip="Clear Formatting"
            >
                <Eraser size={18} />
            </ToolbarButton>
        </div>
    )
}
