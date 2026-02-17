"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion } from "framer-motion"
import {
    House,
    ChartLineUp,
    CurrencyDollar,
    Books,
    Sparkle,
    Gear,
    SignOut,
    CaretLeft,
    Plus,
    Bell,
    User,
    Coin
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

interface DashboardLayoutProps {
    children: React.ReactNode
}

const NAV_ITEMS = [
    { href: "/webtoon/dashboard", icon: House, label: "Overview" },
    { href: "/webtoon/dashboard/analytics", icon: ChartLineUp, label: "Analytics" },
    { href: "/webtoon/dashboard/payments", icon: CurrencyDollar, label: "Payments" },
    { href: "/webtoon/dashboard/series", icon: Books, label: "My Series" },
    { href: "/webtoon/dashboard/ai-tools", icon: Sparkle, label: "AI Tools" },
    { href: "/webtoon/dashboard/settings", icon: Gear, label: "Settings" },
]

export function DashboardLayout({ children }: DashboardLayoutProps) {
    const pathname = usePathname()

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-white flex">
            {/* Sidebar */}
            <aside className="w-64 border-r border-white/5 flex flex-col">
                {/* Logo */}
                <div className="h-16 px-6 flex items-center border-b border-white/5">
                    <Link href="/webtoon" className="flex items-center gap-2 group">
                        <CaretLeft size={16} className="text-white/40 group-hover:text-white transition-colors" />
                        <span className="text-sm text-white/60 group-hover:text-white transition-colors">Back to Toons</span>
                    </Link>
                </div>

                {/* Creator Info */}
                <div className="p-4 border-b border-white/5">
                    <div className="flex items-center gap-3 p-3 rounded-xl bg-white/5">
                        <div className="size-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center font-bold">
                            C
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="font-semibold truncate">Creator Studio</div>
                            <div className="text-xs text-emerald-400">Pro Plan</div>
                        </div>
                    </div>
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 space-y-1">
                    {NAV_ITEMS.map((item) => {
                        const isActive = pathname === item.href ||
                            (item.href !== "/webtoon/dashboard" && pathname?.startsWith(item.href))

                        return (
                            <Link key={item.href} href={item.href}>
                                <div className={cn(
                                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all",
                                    "text-white/60 hover:text-white hover:bg-white/5",
                                    isActive && "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                                )}>
                                    <item.icon size={20} weight={isActive ? "fill" : "regular"} />
                                    <span className="font-medium">{item.label}</span>
                                    {item.label === "AI Tools" && (
                                        <span className="ml-auto px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-400">
                                            NEW
                                        </span>
                                    )}
                                </div>
                            </Link>
                        )
                    })}
                </nav>

                {/* Quick Stats */}
                <div className="p-4 border-t border-white/5">
                    <div className="p-4 rounded-xl bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/20">
                        <div className="flex items-center gap-2 mb-2">
                            <Coin size={16} className="text-amber-400" weight="fill" />
                            <span className="text-xs text-white/50">This Month</span>
                        </div>
                        <div className="text-2xl font-bold text-white">$2,847</div>
                        <div className="text-xs text-emerald-400 mt-1">↑ 23% from last month</div>
                    </div>
                </div>

                {/* Logout */}
                <div className="p-4 border-t border-white/5">
                    <button className="flex items-center gap-3 w-full px-4 py-3 rounded-xl text-white/40 hover:text-white hover:bg-white/5 transition-colors">
                        <SignOut size={20} />
                        <span className="font-medium">Sign Out</span>
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
                {/* Top Bar */}
                <header className="h-16 border-b border-white/5 flex items-center justify-between px-6">
                    <div className="flex items-center gap-4">
                        <h1 className="text-lg font-semibold">Creator Dashboard</h1>
                    </div>

                    <div className="flex items-center gap-3">
                        <Button
                            size="sm"
                            className="bg-emerald-500 hover:bg-emerald-600 text-white"
                        >
                            <Plus size={16} className="mr-2" />
                            New Series
                        </Button>

                        <button className="p-2.5 rounded-xl hover:bg-white/5 transition-colors relative">
                            <Bell size={20} className="text-white/60" />
                            <span className="absolute top-1.5 right-1.5 size-2 bg-emerald-500 rounded-full" />
                        </button>

                        <div className="size-9 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center font-bold cursor-pointer">
                            <User size={18} weight="fill" />
                        </div>
                    </div>
                </header>

                {/* Page Content */}
                <main className="flex-1 overflow-y-auto">
                    {children}
                </main>
            </div>
        </div>
    )
}
