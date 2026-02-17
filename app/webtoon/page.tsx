"use client"

import { motion } from "framer-motion"
import { WebtoonLayout } from "@/app/components/webtoon/webtoon-layout"
import { WebtoonCard } from "@/app/components/webtoon/webtoon-card"
import { Button } from "@/components/ui/button"
import {
    Coin,
    ArrowRight,
    Sparkle,
    Palette,
    ChartLineUp,
    Users,
    Globe,
    Lightning,
    Star,
    Heart,
    Trophy,
    Rocket,
    BookOpen,
    PaintBrush,
    CurrencyDollar,
    CheckCircle
} from "@phosphor-icons/react"
import Link from "next/link"

// Mock data for featured webtoons
const FEATURED_WEBTOONS = [
    {
        id: "1",
        title: "The Dragon's Heir",
        author: "ArtMaster",
        cover: "https://picsum.photos/seed/webtoon1/400/600",
        genre: "Fantasy",
        views: "2.3M",
        likes: "156K",
        rating: 4.9,
        isHot: true
    },
    {
        id: "2",
        title: "Love in Seoul",
        author: "K-Romance",
        cover: "https://picsum.photos/seed/webtoon2/400/600",
        genre: "Romance",
        views: "1.8M",
        likes: "98K",
        rating: 4.7,
        isNew: true
    },
    {
        id: "3",
        title: "Shadow Knight",
        author: "DarkArts",
        cover: "https://picsum.photos/seed/webtoon3/400/600",
        genre: "Action",
        views: "3.1M",
        likes: "201K",
        rating: 4.8,
        isHot: true
    },
    {
        id: "4",
        title: "Campus Mystery",
        author: "ThrillerKing",
        cover: "https://picsum.photos/seed/webtoon4/400/600",
        genre: "Mystery",
        views: "890K",
        likes: "45K",
        rating: 4.5,
        isNew: true
    },
    {
        id: "5",
        title: "Mecha Genesis",
        author: "SciFiPro",
        cover: "https://picsum.photos/seed/webtoon5/400/600",
        genre: "Sci-Fi",
        views: "1.2M",
        likes: "78K",
        rating: 4.6
    },
]

const STATS = [
    { label: "Active Creators", value: "12K+", icon: PaintBrush },
    { label: "Published Series", value: "45K+", icon: BookOpen },
    { label: "Monthly Readers", value: "8M+", icon: Users },
    { label: "Paid to Creators", value: "$2.4M", icon: CurrencyDollar },
]

const HOW_IT_WORKS = [
    {
        step: 1,
        title: "Create Your Series",
        description: "Upload your webtoon panels, set your pricing, and customize your series page.",
        icon: Palette
    },
    {
        step: 2,
        title: "Grow Your Audience",
        description: "Get discovered through our recommendation engine and build your fanbase.",
        icon: ChartLineUp
    },
    {
        step: 3,
        title: "Earn 100% Revenue",
        description: "Every coin purchase goes directly to you. No platform fees, ever.",
        icon: Coin
    },
]

const CREATOR_FEATURES = [
    { text: "100% of coin revenue", icon: Coin },
    { text: "AI-powered 4K upscaling", icon: Sparkle },
    { text: "Instant multi-language translation", icon: Globe },
    { text: "Real-time analytics dashboard", icon: ChartLineUp },
    { text: "Direct fan engagement", icon: Heart },
    { text: "Fast weekly payouts", icon: Lightning },
]

export default function WebtoonLandingPage() {
    return (
        <WebtoonLayout>
            {/* Hero Section */}
            <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
                {/* Background Glow */}
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-[600px] h-[600px] bg-emerald-500/20 rounded-full blur-[150px] animate-pulse" />
                </div>

                <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
                    {/* Badge */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                        className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-8"
                    >
                        <Trophy size={16} className="text-amber-400" weight="fill" />
                        <span className="text-sm font-medium text-emerald-400">Creators keep 100% of coin revenue</span>
                    </motion.div>

                    {/* Main Heading */}
                    <motion.h1
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.1 }}
                        className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-6"
                    >
                        <span className="bg-clip-text text-transparent bg-gradient-to-b from-white via-white/90 to-white/60">
                            Where Stories
                        </span>
                        <br />
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400">
                            Come Alive
                        </span>
                    </motion.h1>

                    {/* Subtitle */}
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                        className="text-xl md:text-2xl text-white/60 max-w-2xl mx-auto mb-10 leading-relaxed"
                    >
                        The premium webtoon platform where{" "}
                        <span className="text-emerald-400 font-semibold">creators earn 100%</span> of reader coin purchases.
                        No platform fees. Just pure creativity.
                    </motion.p>

                    {/* CTAs */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.3 }}
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                    >
                        <Link href="/webtoon/explore">
                            <Button size="lg" className="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white border-0 shadow-lg shadow-emerald-500/25 h-14 px-8 text-lg">
                                Start Reading
                                <ArrowRight size={20} className="ml-2" />
                            </Button>
                        </Link>
                        <Link href="/webtoon/dashboard">
                            <Button size="lg" variant="outline" className="border-white/20 text-white hover:bg-white/5 h-14 px-8 text-lg">
                                <Rocket size={20} className="mr-2" />
                                Become a Creator
                            </Button>
                        </Link>
                    </motion.div>

                    {/* Stats Row */}
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, delay: 0.5 }}
                        className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-20 max-w-4xl mx-auto"
                    >
                        {STATS.map((stat) => (
                            <div key={stat.label} className="text-center">
                                <div className="flex items-center justify-center mb-2">
                                    <stat.icon size={24} className="text-emerald-400" weight="duotone" />
                                </div>
                                <div className="text-3xl font-bold text-white">{stat.value}</div>
                                <div className="text-sm text-white/50">{stat.label}</div>
                            </div>
                        ))}
                    </motion.div>
                </div>

                {/* Scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 1 }}
                    className="absolute bottom-8 left-1/2 -translate-x-1/2"
                >
                    <motion.div
                        animate={{ y: [0, 8, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                        className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-2"
                    >
                        <motion.div className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                    </motion.div>
                </motion.div>
            </section>

            {/* Featured Webtoons */}
            <section className="py-24 px-6">
                <div className="max-w-7xl mx-auto">
                    <div className="flex items-end justify-between mb-12">
                        <div>
                            <h2 className="text-3xl md:text-4xl font-bold mb-3">
                                <span className="text-white">Featured</span>{" "}
                                <span className="text-emerald-400">Stories</span>
                            </h2>
                            <p className="text-white/50">Discover trending webtoons loved by millions</p>
                        </div>
                        <Link href="/webtoon/explore" className="hidden md:flex items-center gap-2 text-emerald-400 hover:text-emerald-300 transition-colors">
                            View All
                            <ArrowRight size={18} />
                        </Link>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
                        {FEATURED_WEBTOONS.map((webtoon, idx) => (
                            <motion.div
                                key={webtoon.id}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ duration: 0.5, delay: idx * 0.1 }}
                            >
                                <WebtoonCard {...webtoon} />
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* 100% Revenue Section */}
            <section className="py-24 px-6 relative overflow-hidden">
                {/* Background */}
                <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/5 via-transparent to-transparent" />

                <div className="relative max-w-6xl mx-auto">
                    <div className="grid lg:grid-cols-2 gap-16 items-center">
                        {/* Left - Content */}
                        <motion.div
                            initial={{ opacity: 0, x: -30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                        >
                            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 mb-6">
                                <Coin size={16} className="text-amber-400" weight="fill" />
                                <span className="text-sm font-medium text-amber-400">Creator First Platform</span>
                            </div>

                            <h2 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
                                <span className="text-white">You Create.</span><br />
                                <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-teal-400">
                                    You Earn 100%.
                                </span>
                            </h2>

                            <p className="text-lg text-white/60 mb-8 leading-relaxed">
                                Unlike other platforms that take up to 50% of your earnings, Ziro Toons gives you
                                <span className="text-white font-semibold"> every single coin</span> your readers spend on your work.
                                Your art, your revenue.
                            </p>

                            <div className="grid grid-cols-2 gap-4 mb-8">
                                {CREATOR_FEATURES.map((feature) => (
                                    <div key={feature.text} className="flex items-center gap-3">
                                        <div className="size-8 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                                            <feature.icon size={16} className="text-emerald-400" />
                                        </div>
                                        <span className="text-sm text-white/80">{feature.text}</span>
                                    </div>
                                ))}
                            </div>

                            <Link href="/webtoon/dashboard">
                                <Button size="lg" className="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white border-0">
                                    Start Creating Today
                                    <ArrowRight size={18} className="ml-2" />
                                </Button>
                            </Link>
                        </motion.div>

                        {/* Right - Visual */}
                        <motion.div
                            initial={{ opacity: 0, x: 30 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            className="relative"
                        >
                            <div className="relative aspect-square max-w-md mx-auto">
                                {/* Glowing circle */}
                                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 to-teal-500/20 rounded-full blur-3xl" />

                                {/* Center coin */}
                                <div className="absolute inset-0 flex items-center justify-center">
                                    <motion.div
                                        animate={{ rotate: 360 }}
                                        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                                        className="relative"
                                    >
                                        <div className="size-48 rounded-full bg-gradient-to-br from-amber-400 to-amber-600 shadow-2xl shadow-amber-500/30 flex items-center justify-center">
                                            <Coin size={80} weight="fill" className="text-amber-900" />
                                        </div>
                                    </motion.div>
                                </div>

                                {/* Floating percentage */}
                                <motion.div
                                    animate={{ y: [-10, 10, -10] }}
                                    transition={{ duration: 3, repeat: Infinity }}
                                    className="absolute top-0 right-0 px-4 py-2 rounded-xl bg-emerald-500 text-white font-bold text-2xl shadow-lg shadow-emerald-500/30"
                                >
                                    100%
                                </motion.div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </section>

            {/* How It Works */}
            <section className="py-24 px-6">
                <div className="max-w-5xl mx-auto">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-4xl font-bold mb-4">
                            <span className="text-white">How It</span>{" "}
                            <span className="text-emerald-400">Works</span>
                        </h2>
                        <p className="text-white/50 max-w-xl mx-auto">
                            Start earning from your creativity in three simple steps
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8">
                        {HOW_IT_WORKS.map((item, idx) => (
                            <motion.div
                                key={item.step}
                                initial={{ opacity: 0, y: 30 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: idx * 0.2 }}
                                className="relative p-8 rounded-2xl bg-white/5 border border-white/10 hover:border-emerald-500/30 transition-colors group"
                            >
                                {/* Step Number */}
                                <div className="absolute -top-4 -left-4 size-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center font-bold text-lg shadow-lg shadow-emerald-500/30">
                                    {item.step}
                                </div>

                                <div className="size-14 rounded-xl bg-emerald-500/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                                    <item.icon size={28} className="text-emerald-400" weight="duotone" />
                                </div>

                                <h3 className="text-xl font-bold text-white mb-3">{item.title}</h3>
                                <p className="text-white/50 leading-relaxed">{item.description}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-24 px-6">
                <div className="max-w-4xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="relative p-12 md:p-16 rounded-3xl overflow-hidden text-center"
                    >
                        {/* Background */}
                        <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/20 via-teal-500/10 to-cyan-500/20" />
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(16,185,129,0.1),transparent_70%)]" />
                        <div className="absolute inset-0 border border-emerald-500/20 rounded-3xl" />

                        <div className="relative z-10">
                            <Sparkle size={48} className="text-emerald-400 mx-auto mb-6" weight="fill" />
                            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                                Ready to Share Your Story?
                            </h2>
                            <p className="text-lg text-white/60 mb-8 max-w-xl mx-auto">
                                Join thousands of creators earning from their passion. Your next chapter starts here.
                            </p>
                            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                                <Link href="/webtoon/dashboard">
                                    <Button size="lg" className="bg-white text-black hover:bg-white/90 h-14 px-8 text-lg font-semibold">
                                        Create Your Series
                                        <ArrowRight size={20} className="ml-2" />
                                    </Button>
                                </Link>
                                <Link href="/webtoon/explore">
                                    <Button size="lg" variant="outline" className="border-white/30 text-white hover:bg-white/10 h-14 px-8 text-lg">
                                        Explore Webtoons
                                    </Button>
                                </Link>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-12 px-6 border-t border-white/5">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
                    <div className="flex items-center gap-3">
                        <div className="size-8 rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                            <BookOpen size={16} weight="fill" className="text-white" />
                        </div>
                        <span className="font-bold">
                            Ziro<span className="text-emerald-400">Toons</span>
                        </span>
                    </div>
                    <div className="flex items-center gap-6 text-sm text-white/40">
                        <span>Terms</span>
                        <span>Privacy</span>
                        <span>Creator Guidelines</span>
                        <span>Help</span>
                    </div>
                    <p className="text-sm text-white/30">© 2026 Ziro Project</p>
                </div>
            </footer>
        </WebtoonLayout>
    )
}
