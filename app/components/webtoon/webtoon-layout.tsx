"use client"

import { useMotionValue, useSpring, useTransform, useMotionTemplate, motion } from "framer-motion"
import { useEffect } from "react"
import Link from "next/link"
import {
    House,
    Compass,
    BookOpen,
    Crown,
    Coin,
    User,
    MagnifyingGlass,
    Bell
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface WebtoonLayoutProps {
    children: React.ReactNode
    showHeader?: boolean
}

export function WebtoonLayout({ children, showHeader = true }: WebtoonLayoutProps) {
    const mouseX = useMotionValue(0)
    const mouseY = useMotionValue(0)

    const springX = useSpring(mouseX, { stiffness: 50, damping: 30 })
    const springY = useSpring(mouseY, { stiffness: 50, damping: 30 })

    const bgX = useTransform(springX, [0, typeof window !== 'undefined' ? window.innerWidth : 1920], ["-2%", "2%"])
    const bgY = useTransform(springY, [0, typeof window !== 'undefined' ? window.innerHeight : 1080], ["-2%", "2%"])

    const maskImage = useMotionTemplate`radial-gradient(600px circle at ${mouseX}px ${mouseY}px, black, transparent)`

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            mouseX.set(e.clientX)
            mouseY.set(e.clientY)
        }
        window.addEventListener("mousemove", handleMouseMove)
        return () => window.removeEventListener("mousemove", handleMouseMove)
    }, [mouseX, mouseY])

    return (
        <div className="min-h-screen w-full bg-[#0a0a0a] text-white overflow-hidden">
            {/* Dynamic Emerald/Teal Gradient Background */}
            <motion.div
                className="pointer-events-none fixed inset-[-5%] z-0 opacity-40"
                style={{ x: bgX, y: bgY }}
            >
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(16,185,129,0.12),transparent_50%)]" />
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_80%,rgba(6,182,212,0.08),transparent_50%)]" />
            </motion.div>

            {/* Interactive Grid */}
            <motion.div
                className="pointer-events-none fixed inset-0 z-0 bg-[linear-gradient(to_right,#80808008_1px,transparent_1px),linear-gradient(to_bottom,#80808008_1px,transparent_1px)] bg-[size:40px_40px]"
                style={{ maskImage }}
            />

            {/* Grainy texture */}
            <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.015] mix-blend-overlay">
                <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] brightness-100 contrast-150" />
            </div>

            {/* Header */}
            {showHeader && (
                <header className="fixed top-0 left-0 right-0 z-50 h-16 bg-[#0a0a0a]/80 backdrop-blur-xl border-b border-white/5">
                    <div className="max-w-7xl mx-auto h-full px-6 flex items-center justify-between">
                        {/* Logo */}
                        <Link href="/webtoon" className="flex items-center gap-3 group">
                            <div className="size-9 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 group-hover:shadow-emerald-500/40 transition-shadow">
                                <BookOpen size={20} weight="fill" className="text-white" />
                            </div>
                            <span className="text-xl font-bold tracking-tight">
                                Ziro<span className="text-emerald-400">Toons</span>
                            </span>
                        </Link>

                        {/* Navigation */}
                        <nav className="hidden md:flex items-center gap-1">
                            <NavLink href="/webtoon" icon={House}>Home</NavLink>
                            <NavLink href="/webtoon/explore" icon={Compass}>Explore</NavLink>
                            <NavLink href="/webtoon/originals" icon={Crown}>Originals</NavLink>
                        </nav>

                        {/* Right Section */}
                        <div className="flex items-center gap-3">
                            {/* Search */}
                            <button className="p-2.5 rounded-xl hover:bg-white/5 transition-colors">
                                <MagnifyingGlass size={20} className="text-white/60" />
                            </button>

                            {/* Notifications */}
                            <button className="p-2.5 rounded-xl hover:bg-white/5 transition-colors relative">
                                <Bell size={20} className="text-white/60" />
                                <span className="absolute top-1.5 right-1.5 size-2 bg-emerald-500 rounded-full" />
                            </button>

                            {/* Coins */}
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20">
                                <Coin size={18} className="text-amber-400" weight="fill" />
                                <span className="text-sm font-semibold text-amber-400">250</span>
                            </div>

                            {/* Sign In / User */}
                            <Button
                                variant="outline"
                                size="sm"
                                className="bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 hover:text-emerald-300"
                            >
                                <User size={16} className="mr-2" />
                                Sign In
                            </Button>
                        </div>
                    </div>
                </header>
            )}

            {/* Main Content */}
            <main className={cn("relative z-10", showHeader && "pt-16")}>
                {children}
            </main>
        </div>
    )
}

function NavLink({ href, icon: Icon, children }: { href: string; icon: React.ElementType; children: React.ReactNode }) {
    return (
        <Link
            href={href}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-white/60 hover:text-white hover:bg-white/5 transition-colors text-sm font-medium"
        >
            <Icon size={18} />
            {children}
        </Link>
    )
}
