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
    SidebarGroupContent
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
    Hash
} from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

const MOCK_PAGES = [
    { id: "demo", title: "Product Requirements", icon: Files },
    { id: "marketing", title: "Marketing Strategy", icon: BookOpen },
    { id: "team", title: "Team Goals 2026", icon: Users },
    { id: "ideas", title: "Content Ideas", icon: Star },
]

export function NotionSidebar() {
    const router = useRouter()
    const pathname = usePathname()

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
                    <span className="font-semibold text-sm tracking-tight">Zenchi Notes</span>
                </div>
                <Button variant="ghost" size="icon" className="size-7" onClick={() => router.push("/")}>
                    <CaretLeft size={14} />
                </Button>
            </SidebarHeader>

            {/* Content */}
            <SidebarContent className="p-2">

                {/* Quick Actions */}
                <SidebarMenu className="mb-4">
                    <SidebarMenuItem>
                        <SidebarMenuButton
                            onClick={() => { }} // Search trigger 
                            className="text-muted-foreground hover:text-foreground"
                        >
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

                {/* Workspaces */}
                <SidebarGroup className="mt-2">
                    <SidebarGroupLabel>Private</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {MOCK_PAGES.map((page) => (
                                <SidebarMenuItem key={page.id}>
                                    <SidebarMenuButton
                                        onClick={() => router.push(`/notion/${page.id === 'demo' ? 'demo' : '#'}`)}
                                        isActive={pathname.includes(page.id)}
                                        className="group/item"
                                    >
                                        <CaretRight size={12} className="text-muted-foreground/50 transition-transform group-hover/item:text-foreground group-data-[state=open]/collapsible:rotate-90" />
                                        <page.icon size={16} className="text-muted-foreground" />
                                        <span>{page.title}</span>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}

                            <SidebarMenuItem>
                                <SidebarMenuButton className="text-muted-foreground opacity-50 hover:opacity-100">
                                    <Plus size={14} />
                                    <span>Add a page</span>
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
