"use client"

import { usePathname, useRouter } from "next/navigation"
import {
    Sidebar,
    SidebarContent,
    SidebarHeader,
    SidebarMenu,
    SidebarMenuItem,
    SidebarMenuButton,
    SidebarGroup,
    SidebarGroupLabel,
    SidebarGroupContent,
    SidebarMenuSub,
    SidebarMenuSubItem,
    SidebarMenuSubButton,
    useSidebar
} from "@/components/ui/sidebar"
import {
    CaretLeft,
    Files,
    Plus,
    Gear,
    MagnifyingGlass,
    Star,
    Clock,
    Users,
    CaretRight,
    BookOpen,
    Hash,
    FolderOpen
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { useState } from "react"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"

const MOCK_PROJECTS = [
    {
        id: "marketing-q1",
        title: "Marketing Q1",
        icon: FolderOpen,
        pages: [
            { id: "strategy", title: "Strategy Doc" },
            { id: "assets", title: "Brand Assets" },
            { id: "campaign", title: "Social Campaign" }
        ]
    },
    {
        id: "product-dev",
        title: "Product Dev",
        icon: FolderOpen,
        pages: [
            { id: "reqs", title: "Requirements" },
            { id: "roadmap", title: "Roadmap 2026" }
        ]
    },
]

export function NotionSidebar() {
    const router = useRouter()
    const pathname = usePathname()

    // Track open states for projects
    const [openProjects, setOpenProjects] = useState<Record<string, boolean>>({
        "marketing-q1": true,
        "product-dev": false
    })

    const toggleProject = (id: string) => {
        setOpenProjects(prev => ({ ...prev, [id]: !prev[id] }))
    }

    return (
        <Sidebar collapsible="none" className="h-full border-r border-border/40 bg-sidebar/50 backdrop-blur-xl w-64 hidden md:flex">
            {/* Header */}
            <SidebarHeader className="p-4 border-b border-border/40 h-14 flex flex-row items-center justify-between">
                <div
                    className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
                    onClick={() => router.push("/")}
                >
                    <div className="size-6 rounded-md bg-blue-600/20 text-blue-500 flex items-center justify-center border border-blue-500/20">
                        <BookOpen size={14} weight="fill" />
                    </div>
                    <span className="font-semibold text-sm tracking-tight">Ziro Notes</span>
                </div>
                <Button variant="ghost" size="icon" className="size-7" onClick={() => router.push("/")}>
                    <CaretLeft size={14} />
                </Button>
            </SidebarHeader>

            {/* Content */}
            <SidebarContent className="p-2">

                {/* Quick Actions */}
                <SidebarMenu className="mb-4">
                    {/* ... existing Search & New Page items ... */}
                    <SidebarMenuItem>
                        <SidebarMenuButton className="text-muted-foreground hover:text-foreground">
                            <MagnifyingGlass size={16} />
                            <span>Search</span>
                            <kbd className="ml-auto pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
                                <span className="text-xs">⌘</span>K
                            </kbd>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                    <SidebarMenuItem>
                        <SidebarMenuButton onClick={() => router.push("/notion/demo")} className="text-muted-foreground hover:text-foreground">
                            <Plus size={16} />
                            <span>New Page</span>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                </SidebarMenu>

                {/* Favorites */}
                <SidebarGroup>
                    <SidebarGroupLabel>Favorites</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            <SidebarMenuItem>
                                <SidebarMenuButton className="justify-start">
                                    <Clock size={16} className="text-muted-foreground" />
                                    <span>Recent</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                            <SidebarMenuItem>
                                <SidebarMenuButton className="justify-start">
                                    <Star size={16} className="text-yellow-500/70" />
                                    <span>Important</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                {/* Projects (Collapsible) */}
                <SidebarGroup className="mt-2">
                    <SidebarGroupLabel>Projects</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {MOCK_PROJECTS.map((project) => (
                                <Collapsible
                                    key={project.id}
                                    open={openProjects[project.id]}
                                    onOpenChange={() => toggleProject(project.id)}
                                    className="group/collapsible"
                                >
                                    <SidebarMenuItem>
                                        <CollapsibleTrigger asChild>
                                            <SidebarMenuButton className="group/item w-full justify-start">
                                                <CaretRight
                                                    size={12}
                                                    className="text-muted-foreground/50 transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90"
                                                />
                                                <project.icon size={16} className="text-muted-foreground" />
                                                <span>{project.title}</span>
                                            </SidebarMenuButton>
                                        </CollapsibleTrigger>

                                        <CollapsibleContent>
                                            <SidebarMenuSub>
                                                {project.pages.map(page => (
                                                    <SidebarMenuSubItem key={page.id}>
                                                        <SidebarMenuSubButton
                                                            onClick={() => router.push(`/notion/demo`)} // Redirects to demo for now
                                                            isActive={false}
                                                        >
                                                            {page.title}
                                                        </SidebarMenuSubButton>
                                                    </SidebarMenuSubItem>
                                                ))}
                                            </SidebarMenuSub>
                                        </CollapsibleContent>
                                    </SidebarMenuItem>
                                </Collapsible>
                            ))}

                            <SidebarMenuItem>
                                <SidebarMenuButton className="text-muted-foreground opacity-50 hover:opacity-100">
                                    <Plus size={14} />
                                    <span>Add a project</span>
                                </SidebarMenuButton>
                            </SidebarMenuItem>
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

            </SidebarContent>

            {/* Footer */}
            <div className="mt-auto p-4 border-t border-border/40">
                <Button variant="ghost" className="w-full justify-start gap-2 text-muted-foreground">
                    <Gear size={16} />
                    Settings
                </Button>
            </div>
        </Sidebar>
    )
}
