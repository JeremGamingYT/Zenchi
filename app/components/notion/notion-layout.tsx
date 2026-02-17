"use client"

import { NotionSidebar } from "@/app/components/notion/notion-sidebar"
import { SidebarProvider } from "@/components/ui/sidebar"
import { useMotionValue, useSpring, useTransform, useMotionTemplate, motion } from "framer-motion"
import { useEffect } from "react"

export function NotionLayout({ children }: { children: React.ReactNode }) {
    // Mouse position for interactive effects
    const mouseX = useMotionValue(0)
    const mouseY = useMotionValue(0)

    const springX = useSpring(mouseX, { stiffness: 50, damping: 30 })
    const springY = useSpring(mouseY, { stiffness: 50, damping: 30 })

    const bgX = useTransform(springX, [0, typeof window !== 'undefined' ? window.innerWidth : 1920], ["-3%", "3%"])
    const bgY = useTransform(springY, [0, typeof window !== 'undefined' ? window.innerHeight : 1080], ["-3%", "3%"])

    const maskImage = useMotionTemplate`radial-gradient(700px circle at ${mouseX}px ${mouseY}px, black, transparent)`

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            mouseX.set(e.clientX)
            mouseY.set(e.clientY)
        }
        window.addEventListener("mousemove", handleMouseMove)
        return () => window.removeEventListener("mousemove", handleMouseMove)
    }, [mouseX, mouseY])

    return (
        <SidebarProvider defaultOpen>
            <div className="bg-background relative flex h-dvh w-full overflow-hidden">

                {/* Dynamic Blue/Violet Gradient Background */}
                <motion.div
                    className="pointer-events-none fixed inset-[-5%] z-0 opacity-40"
                    style={{ x: bgX, y: bgY }}
                >
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_30%,rgba(59,130,246,0.1),transparent_50%)]" />
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_70%,rgba(139,92,246,0.08),transparent_50%)]" />
                </motion.div>

                {/* Interactive Grid Overlay */}
                <motion.div
                    className="pointer-events-none fixed inset-0 z-0 bg-[linear-gradient(to_right,#80808010_1px,transparent_1px),linear-gradient(to_bottom,#80808010_1px,transparent_1px)] bg-[size:28px_28px]"
                    style={{ maskImage }}
                />

                {/* Static dim grid */}
                <div className="pointer-events-none fixed inset-0 z-0 bg-[linear-gradient(to_right,#80808005_1px,transparent_1px),linear-gradient(to_bottom,#80808005_1px,transparent_1px)] bg-[size:28px_28px] [mask-image:radial-gradient(ellipse_70%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

                {/* Grainy texture */}
                <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.02] mix-blend-overlay">
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
