"use client"

import { NotionSidebar } from "@/app/components/notion/notion-sidebar"
import { SidebarProvider } from "@/components/ui/sidebar"

export function NotionLayout({ children }: { children: React.ReactNode }) {
    return (
        <SidebarProvider defaultOpen>
            <div className="bg-background relative flex h-dvh w-full overflow-hidden">
                {/* Global Grainy Background Reuse */}
                <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.03] mix-blend-overlay">
                    <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] brightness-100 contrast-150" />
                </div>

                <NotionSidebar />

                <main className="flex-1 relative z-10 h-full overflow-y-auto overflow-x-hidden">
                    {children}
                </main>
            </div>
        </SidebarProvider>
    )
}
