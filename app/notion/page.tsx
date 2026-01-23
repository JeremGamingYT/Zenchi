"use client"

import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import {
    Plus,
    Clock,
    SquaresFour,
    ListBullets,
    FileText,
    DotsThree,
    MagnifyingGlass
} from "@phosphor-icons/react"
import { useRouter } from "next/navigation"
import { NotionLayout } from "@/app/components/notion/notion-layout"

const RECENT_DOCS = [
    { id: 1, title: "Product Requirements v2", date: "Edited 2h ago", icon: FileText, color: "text-blue-500" },
    { id: 2, title: "Q1 Marketing Plan", date: "Edited yesterday", icon: FileText, color: "text-purple-500" },
    { id: 3, title: "Team Retrospective", date: "Edited 2 days ago", icon: FileText, color: "text-rose-500" },
]

export default function NotesPage() {
    const router = useRouter()
    const currentHour = new Date().getHours()
    let greeting = "Good evening"
    if (currentHour < 12) greeting = "Good morning"
    else if (currentHour < 18) greeting = "Good afternoon"

    return (
        <NotionLayout>
            <div className="min-h-full w-full p-8 md:p-12 max-w-6xl mx-auto flex flex-col gap-12">

                {/* Hero Header */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-forwards">
                    <div className="space-y-2">
                        <h1 className="text-4xl font-bold tracking-tight">{greeting}, Creator.</h1>
                        <p className="text-muted-foreground text-lg">Ready to organize your thoughts?</p>
                    </div>
                    <div className="flex gap-3">
                        <Button variant="outline" className="gap-2 bg-background/50 backdrop-blur-sm">
                            <MagnifyingGlass size={16} /> Search
                        </Button>
                        <Button onClick={() => router.push("/notion/demo")} className="gap-2 bg-blue-600 hover:bg-blue-700 text-white border-0 shadow-lg shadow-blue-900/20">
                            <Plus size={16} weight="bold" /> New Page
                        </Button>
                    </div>
                </div>

                {/* Quick Access / Recents */}
                <div className="space-y-6 opacity-0 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-100 fill-mode-forwards">
                    <div className="flex items-center justify-between border-b border-border/40 pb-4">
                        <h2 className="text-xl font-semibold flex items-center gap-2">
                            <Clock size={20} className="text-muted-foreground" />
                            Recent Pages
                        </h2>
                        <div className="flex gap-1">
                            <Button variant="ghost" size="icon" className="h-8 w-8"><SquaresFour size={16} /></Button>
                            <Button variant="ghost" size="icon" className="h-8 w-8"><ListBullets size={16} /></Button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {/* New Page Card */}
                        <motion.div
                            whileHover={{ y: -4 }}
                            className="group flex flex-col justify-center items-center h-48 rounded-xl border-2 border-dashed border-border/60 hover:border-blue-500/50 hover:bg-blue-500/5 transition-all cursor-pointer bg-background/30 backdrop-blur-sm"
                            onClick={() => router.push("/notion/demo")}
                        >
                            <div className="size-12 rounded-full bg-blue-500/10 text-blue-500 flex items-center justify-center mb-3 group-hover:scale-110 transition-transform">
                                <Plus size={24} weight="bold" />
                            </div>
                            <span className="font-medium text-muted-foreground group-hover:text-foreground transition-colors">Create New Page</span>
                        </motion.div>

                        {/* Recent Doc Cards */}
                        {RECENT_DOCS.map((doc, idx) => (
                            <motion.div
                                key={doc.id}
                                whileHover={{ y: -4 }}
                                className="group relative flex flex-col h-48 rounded-xl border border-border/50 bg-card/50 backdrop-blur-md overflow-hidden hover:shadow-xl transition-all cursor-pointer"
                                onClick={() => router.push("/notion/demo")}
                            >
                                {/* Card Header Color */}
                                <div className={`h-2 w-full bg-gradient-to-r ${idx === 0 ? "from-blue-500 to-cyan-500" : idx === 1 ? "from-purple-500 to-pink-500" : "from-rose-500 to-orange-500"} opacity-70`} />

                                <div className="p-5 flex flex-col h-full">
                                    <div className="flex justify-between items-start">
                                        <doc.icon size={24} className={doc.color} weight="duotone" />
                                        <Button variant="ghost" size="icon" className="h-6 w-6 -mr-2 -mt-2 opacity-0 group-hover:opacity-100 transition-opacity"><DotsThree size={16} /></Button>
                                    </div>

                                    <div className="mt-auto">
                                        <h3 className="font-semibold text-lg leading-tight mb-1 group-hover:text-blue-500 transition-colors">{doc.title}</h3>
                                        <span className="text-xs text-muted-foreground">{doc.date}</span>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>

                {/* Templates Section */}
                <div className="space-y-6 opacity-0 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-200 fill-mode-forwards">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                        Suggested Templates
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {['Project Roadmap', 'Meeting Notes', 'Weekly Agenda'].map((template) => (
                            <div key={template} className="p-4 rounded-lg border border-border/50 bg-sidebar/30 hover:bg-sidebar/60 transition-colors cursor-pointer flex items-center justify-between group">
                                <span className="font-medium">{template}</span>
                                <span className="text-xs text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity bg-blue-500/10 px-2 py-1 rounded">Use</span>
                            </div>
                        ))}
                    </div>
                </div>

            </div>
        </NotionLayout>
    )
}
