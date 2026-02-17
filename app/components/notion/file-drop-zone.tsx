"use client"

import { useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
    UploadSimple,
    FilePdf,
    FileDoc,
    FileImage,
    FileText,
    X,
    Check
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"

interface UploadedFile {
    id: string
    name: string
    type: string
    size: number
    preview?: string
}

interface FileDropZoneProps {
    onFilesAdded?: (files: UploadedFile[]) => void
    className?: string
}

const FILE_ICONS: Record<string, typeof FilePdf> = {
    'application/pdf': FilePdf,
    'application/msword': FileDoc,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileDoc,
    'image/png': FileImage,
    'image/jpeg': FileImage,
    'image/gif': FileImage,
    'image/webp': FileImage,
}

function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export function FileDropZone({ onFilesAdded, className }: FileDropZoneProps) {
    const [isDragging, setIsDragging] = useState(false)
    const [files, setFiles] = useState<UploadedFile[]>([])

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(true)
    }, [])

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }, [])

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)

        const droppedFiles = Array.from(e.dataTransfer.files)
        const newFiles: UploadedFile[] = droppedFiles.map(file => ({
            id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            name: file.name,
            type: file.type,
            size: file.size,
            preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
        }))

        setFiles(prev => [...prev, ...newFiles])
        onFilesAdded?.(newFiles)
    }, [onFilesAdded])

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFiles = e.target.files
        if (!selectedFiles) return

        const newFiles: UploadedFile[] = Array.from(selectedFiles).map(file => ({
            id: `file-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            name: file.name,
            type: file.type,
            size: file.size,
            preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : undefined
        }))

        setFiles(prev => [...prev, ...newFiles])
        onFilesAdded?.(newFiles)
    }, [onFilesAdded])

    const removeFile = useCallback((id: string) => {
        setFiles(prev => prev.filter(f => f.id !== id))
    }, [])

    const getFileIcon = (type: string) => {
        return FILE_ICONS[type] || FileText
    }

    return (
        <div className={cn("space-y-4", className)}>
            {/* Drop Zone */}
            <motion.div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                animate={{
                    scale: isDragging ? 1.02 : 1,
                    borderColor: isDragging ? 'rgba(59, 130, 246, 0.5)' : 'rgba(255, 255, 255, 0.1)',
                }}
                transition={{ type: "spring", bounce: 0.2, duration: 0.3 }}
                className={cn(
                    "relative group rounded-xl border-2 border-dashed p-8 text-center cursor-pointer transition-all",
                    "bg-gradient-to-b from-blue-500/5 to-violet-500/5",
                    "hover:border-blue-500/30 hover:bg-blue-500/5",
                    isDragging && "border-blue-500/50 bg-blue-500/10"
                )}
            >
                <input
                    type="file"
                    multiple
                    onChange={handleFileInput}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    accept=".pdf,.doc,.docx,.txt,.md,.png,.jpg,.jpeg,.gif,.webp"
                />

                <motion.div
                    animate={{ y: isDragging ? -4 : 0 }}
                    transition={{ type: "spring", bounce: 0.3 }}
                    className="flex flex-col items-center gap-3"
                >
                    <div className={cn(
                        "size-12 rounded-xl flex items-center justify-center transition-colors",
                        "bg-blue-500/10 text-blue-400",
                        isDragging && "bg-blue-500/20 text-blue-300"
                    )}>
                        <UploadSimple size={24} weight="duotone" />
                    </div>
                    <div className="space-y-1">
                        <p className="text-sm font-medium text-foreground/80">
                            {isDragging ? "Drop files here" : "Drag & drop files"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                            PDF, Word, Images, Markdown
                        </p>
                    </div>
                </motion.div>

                {/* Animated border glow when dragging */}
                <AnimatePresence>
                    {isDragging && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 -z-10 rounded-xl bg-blue-500/20 blur-xl"
                        />
                    )}
                </AnimatePresence>
            </motion.div>

            {/* File List */}
            <AnimatePresence mode="popLayout">
                {files.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-2 overflow-hidden"
                    >
                        {files.map((file) => {
                            const Icon = getFileIcon(file.type)
                            return (
                                <motion.div
                                    key={file.id}
                                    layout
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: 20, scale: 0.9 }}
                                    transition={{ type: "spring", bounce: 0.2 }}
                                    className="group flex items-center gap-3 p-3 rounded-lg bg-card/50 border border-border/50 hover:border-border transition-colors"
                                >
                                    {/* Preview or Icon */}
                                    {file.preview ? (
                                        <div className="size-10 rounded-md overflow-hidden bg-muted">
                                            <img
                                                src={file.preview}
                                                alt={file.name}
                                                className="w-full h-full object-cover"
                                            />
                                        </div>
                                    ) : (
                                        <div className={cn(
                                            "size-10 rounded-md flex items-center justify-center",
                                            file.type.includes('pdf') && "bg-red-500/10 text-red-400",
                                            file.type.includes('word') && "bg-blue-500/10 text-blue-400",
                                            !file.type.includes('pdf') && !file.type.includes('word') && "bg-muted text-muted-foreground"
                                        )}>
                                            <Icon size={20} weight="duotone" />
                                        </div>
                                    )}

                                    {/* File Info */}
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium truncate">{file.name}</p>
                                        <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                                    </div>

                                    {/* Actions */}
                                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <motion.button
                                            whileHover={{ scale: 1.1 }}
                                            whileTap={{ scale: 0.9 }}
                                            onClick={() => removeFile(file.id)}
                                            className="p-1.5 rounded-md hover:bg-red-500/10 text-muted-foreground hover:text-red-400 transition-colors"
                                        >
                                            <X size={14} />
                                        </motion.button>
                                    </div>

                                    {/* Uploaded indicator */}
                                    <div className="size-6 rounded-full bg-green-500/10 text-green-400 flex items-center justify-center">
                                        <Check size={12} weight="bold" />
                                    </div>
                                </motion.div>
                            )
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
