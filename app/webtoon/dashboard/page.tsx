"use client"

import { motion } from "framer-motion"
import { DashboardLayout } from "@/app/components/webtoon/dashboard-layout"
import {
    Coin,
    Users,
    Eye,
    Heart,
    TrendUp,
    TrendDown,
    ArrowRight,
    ChartLine,
    CalendarBlank,
    Clock,
    Star,
    Books
} from "@phosphor-icons/react"
import Link from "next/link"
import { Button } from "@/components/ui/button"

const STATS = [
    {
        label: "Total Earnings",
        value: "$12,847",
        change: "+23.5%",
        trend: "up",
        icon: Coin,
        color: "amber"
    },
    {
        label: "Subscribers",
        value: "8,429",
        change: "+12.3%",
        trend: "up",
        icon: Users,
        color: "emerald"
    },
    {
        label: "Total Views",
        value: "2.4M",
        change: "+45.2%",
        trend: "up",
        icon: Eye,
        color: "blue"
    },
    {
        label: "Total Likes",
        value: "156K",
        change: "+8.7%",
        trend: "up",
        icon: Heart,
        color: "rose"
    },
]

const RECENT_TRANSACTIONS = [
    { id: 1, type: "Episode Unlock", title: "Dragon's Heir - Ep.24", amount: 15, time: "2 min ago" },
    { id: 2, type: "Tip", title: "@webtoon_fan_123", amount: 50, time: "15 min ago" },
    { id: 3, type: "Episode Unlock", title: "Dragon's Heir - Ep.23", amount: 15, time: "1 hour ago" },
    { id: 4, type: "Subscription", title: "New monthly subscriber", amount: 100, time: "2 hours ago" },
    { id: 5, type: "Episode Unlock", title: "Shadow Knight - Ep.12", amount: 15, time: "3 hours ago" },
]

const MY_SERIES = [
    { id: "1", title: "The Dragon's Heir", episodes: 24, views: "1.2M", status: "ongoing" },
    { id: "2", title: "Shadow Knight", episodes: 12, views: "890K", status: "ongoing" },
    { id: "3", title: "Love in Spring", episodes: 45, views: "2.1M", status: "completed" },
]

// Simple chart data visualization
const CHART_DATA = [35, 45, 52, 48, 61, 58, 75, 82, 68, 79, 95, 110]
const CHART_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

function MiniChart() {
    const maxValue = Math.max(...CHART_DATA)

    return (
        <div className="flex items-end gap-1 h-32">
            {CHART_DATA.map((value, idx) => (
                <motion.div
                    key={idx}
                    initial={{ height: 0 }}
                    animate={{ height: `${(value / maxValue) * 100}%` }}
                    transition={{ delay: idx * 0.05, duration: 0.5 }}
                    className="flex-1 bg-gradient-to-t from-emerald-500/60 to-emerald-400/40 rounded-t-sm hover:from-emerald-500 hover:to-emerald-400 transition-colors cursor-pointer group relative"
                >
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 px-2 py-1 rounded bg-black text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                        ${value * 10}
                    </div>
                </motion.div>
            ))}
        </div>
    )
}

export default function DashboardPage() {
    return (
        <DashboardLayout>
            <div className="p-6 space-y-8">
                {/* Welcome Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold mb-1">Welcome back, Creator!</h1>
                        <p className="text-white/50">Here's what's happening with your webtoons</p>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-white/40">
                        <CalendarBlank size={16} />
                        <span>January 23, 2026</span>
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {STATS.map((stat, idx) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: idx * 0.1 }}
                            className="p-6 rounded-2xl bg-white/5 border border-white/10 hover:border-white/20 transition-colors"
                        >
                            <div className="flex items-start justify-between mb-4">
                                <div className={`size-12 rounded-xl bg-${stat.color}-500/10 flex items-center justify-center`}>
                                    <stat.icon size={24} className={`text-${stat.color}-400`} weight="duotone" />
                                </div>
                                <div className={`flex items-center gap-1 text-sm font-medium ${stat.trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {stat.trend === 'up' ? <TrendUp size={16} /> : <TrendDown size={16} />}
                                    {stat.change}
                                </div>
                            </div>
                            <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                            <div className="text-sm text-white/50">{stat.label}</div>
                        </motion.div>
                    ))}
                </div>

                {/* Charts Row */}
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Revenue Chart */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="lg:col-span-2 p-6 rounded-2xl bg-white/5 border border-white/10"
                    >
                        <div className="flex items-center justify-between mb-6">
                            <div>
                                <h3 className="font-semibold text-white mb-1">Revenue Overview</h3>
                                <p className="text-sm text-white/40">Monthly earnings this year</p>
                            </div>
                            <div className="flex items-center gap-2">
                                <Button variant="ghost" size="sm" className="text-white/60">
                                    7D
                                </Button>
                                <Button variant="ghost" size="sm" className="text-white/60">
                                    30D
                                </Button>
                                <Button variant="ghost" size="sm" className="bg-white/10 text-white">
                                    12M
                                </Button>
                            </div>
                        </div>

                        <MiniChart />

                        <div className="flex justify-between mt-4 text-xs text-white/30">
                            {CHART_LABELS.map((label) => (
                                <span key={label}>{label}</span>
                            ))}
                        </div>
                    </motion.div>

                    {/* Recent Transactions */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="p-6 rounded-2xl bg-white/5 border border-white/10"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="font-semibold text-white">Recent Earnings</h3>
                            <Link href="/webtoon/dashboard/payments" className="text-xs text-emerald-400 hover:text-emerald-300">
                                View All
                            </Link>
                        </div>

                        <div className="space-y-3">
                            {RECENT_TRANSACTIONS.map((tx) => (
                                <div key={tx.id} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                                    <div className="flex items-center gap-3">
                                        <div className="size-8 rounded-lg bg-amber-500/10 flex items-center justify-center">
                                            <Coin size={14} className="text-amber-400" weight="fill" />
                                        </div>
                                        <div>
                                            <div className="text-sm font-medium text-white truncate max-w-[120px]">{tx.title}</div>
                                            <div className="text-[10px] text-white/40">{tx.type}</div>
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm font-semibold text-emerald-400">+{tx.amount}c</div>
                                        <div className="text-[10px] text-white/30">{tx.time}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                </div>

                {/* My Series */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="p-6 rounded-2xl bg-white/5 border border-white/10"
                >
                    <div className="flex items-center justify-between mb-6">
                        <div>
                            <h3 className="font-semibold text-white mb-1">My Series</h3>
                            <p className="text-sm text-white/40">Manage and track your webtoons</p>
                        </div>
                        <Link href="/webtoon/dashboard/series">
                            <Button variant="outline" size="sm" className="border-white/20 text-white hover:bg-white/5">
                                <Books size={16} className="mr-2" />
                                View All
                            </Button>
                        </Link>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="text-left text-xs text-white/40 border-b border-white/10">
                                    <th className="pb-3 font-medium">Series</th>
                                    <th className="pb-3 font-medium">Episodes</th>
                                    <th className="pb-3 font-medium">Total Views</th>
                                    <th className="pb-3 font-medium">Status</th>
                                    <th className="pb-3 font-medium text-right">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {MY_SERIES.map((series) => (
                                    <tr key={series.id} className="border-b border-white/5 last:border-0">
                                        <td className="py-4">
                                            <div className="flex items-center gap-3">
                                                <div className="size-10 rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 flex items-center justify-center">
                                                    <Books size={18} className="text-emerald-400" />
                                                </div>
                                                <span className="font-medium text-white">{series.title}</span>
                                            </div>
                                        </td>
                                        <td className="py-4 text-white/60">{series.episodes} eps</td>
                                        <td className="py-4 text-white/60">{series.views}</td>
                                        <td className="py-4">
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${series.status === 'ongoing'
                                                    ? 'bg-emerald-500/10 text-emerald-400'
                                                    : 'bg-blue-500/10 text-blue-400'
                                                }`}>
                                                {series.status}
                                            </span>
                                        </td>
                                        <td className="py-4 text-right">
                                            <Button variant="ghost" size="sm" className="text-white/60 hover:text-white">
                                                Manage
                                                <ArrowRight size={14} className="ml-1" />
                                            </Button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>
            </div>
        </DashboardLayout>
    )
}
