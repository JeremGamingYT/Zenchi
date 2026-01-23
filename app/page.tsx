"use client"

import { motion, useMotionValue, useSpring, useTransform } from "framer-motion"
import { useRouter } from "next/navigation"
import {
  Sparkle,
  BookOpen,
  MonitorPlay,
  FilmStrip,
  ArrowRight,
  Users,
  CurrencyDollar,
  Pulse,
  Planet
} from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { useState, useEffect } from "react"

// --- Configuration ---
const stats = [
  { id: 1, label: "Active Users", value: "12,450", icon: Users },
  { id: 2, label: "Creator Payouts", value: "$845k", icon: CurrencyDollar },
  { id: 3, label: "Daily Interactions", value: "1.2M", icon: Pulse },
]

const universes = [
  {
    id: "chat",
    name: "Zenchi Chat",
    tagline: "Intelligence Unbound",
    description: "Access the world's most powerful AI models in one unified interface.",
    icon: Sparkle,
    color: "violet",
    gradient: "from-violet-500/20 via-violet-500/5 to-transparent",
    borderHover: "group-hover:border-violet-500/50",
    textHover: "group-hover:text-violet-400",
    bgHover: "group-hover:bg-violet-950/30",
    route: "/chat",
  },
  {
    id: "notion",
    name: "Zenchi Notes",
    tagline: "Thought Organization",
    description: "A second brain for your ideas, seamlessly integrated with AI.",
    icon: BookOpen,
    color: "blue",
    gradient: "from-blue-500/20 via-blue-500/5 to-transparent",
    borderHover: "group-hover:border-blue-500/50",
    textHover: "group-hover:text-blue-400",
    bgHover: "group-hover:bg-blue-950/30",
    route: "/notion",
  },
  {
    id: "webtoon",
    name: "Zenchi Toons",
    tagline: "Visual Stories",
    description: "Immersive vertical scrolling comics from top creators.",
    icon: MonitorPlay,
    color: "emerald",
    gradient: "from-emerald-500/20 via-emerald-500/5 to-transparent",
    borderHover: "group-hover:border-emerald-500/50",
    textHover: "group-hover:text-emerald-400",
    bgHover: "group-hover:bg-emerald-950/30",
    route: "/webtoon",
  },
  {
    id: "anime",
    name: "Zenchi Watch",
    tagline: "Cinematic Experience",
    description: "Stream your favorite anime in stunning 4K resolution.",
    icon: FilmStrip,
    color: "rose",
    gradient: "from-rose-500/20 via-rose-500/5 to-transparent",
    borderHover: "group-hover:border-rose-500/50",
    textHover: "group-hover:text-rose-400",
    bgHover: "group-hover:bg-rose-950/30",
    route: "/anime",
  },
]

// --- Components ---

function Counter({ value }: { value: string }) {
  // Simple fade-in for numbers for now, complexity relies on preserving the "k" and "M"
  return (
    <span className="tabular-nums tracking-tight">
      {value}
    </span>
  )
}

export default function LandingPage() {
  const router = useRouter()
  const [hoveredUniverse, setHoveredUniverse] = useState<string | null>(null)

  // Mouse position effect for subtle background movement
  const mouseX = useMotionValue(0)
  const mouseY = useMotionValue(0)

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX)
      mouseY.set(e.clientY)
    }
    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [mouseX, mouseY])

  const springX = useSpring(mouseX, { stiffness: 50, damping: 20 })
  const springY = useSpring(mouseY, { stiffness: 50, damping: 20 })

  const bgX = useTransform(springX, [0, typeof window !== 'undefined' ? window.innerWidth : 1000], ["-5%", "5%"])
  const bgY = useTransform(springY, [0, typeof window !== 'undefined' ? window.innerHeight : 1000], ["-5%", "5%"])

  return (
    <div className="relative min-h-screen w-full bg-black text-white selection:bg-white/20 overflow-hidden font-sans flex flex-col">

      {/* Dynamic Background Pattern */}
      <motion.div
        className="fixed inset-[-10%] z-0 pointer-events-none opacity-40 mix-blend-screen"
        style={{ x: bgX, y: bgY }}
      >
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(76,29,149,0.15),transparent_50%)]" />
        <div className="absolute top-0 left-0 w-full h-full bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-100 contrast-150" />
      </motion.div>

      {/* Grid Pattern Overlay */}
      <div className="fixed inset-0 z-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      {/* Header */}
      <header className="relative z-50 w-full p-6 flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="size-8 rounded-lg bg-white/10 flex items-center justify-center border border-white/10 backdrop-blur-md">
            <Planet weight="fill" className="text-white" size={18} />
          </div>
          <span className="font-bold text-lg tracking-tight">Zenchi</span>
        </div>
        <div className="flex items-center gap-4">
          <button className="text-sm text-white/60 hover:text-white transition-colors">About</button>
          <button className="text-sm text-white/60 hover:text-white transition-colors">Creators</button>
          <button
            onClick={() => router.push("/auth")}
            className="px-4 py-2 rounded-full bg-white/10 hover:bg-white/20 border border-white/5 text-sm font-medium transition-all backdrop-blur-sm"
          >
            Sign In
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col justify-center items-center px-4 py-12 max-w-7xl mx-auto w-full">

        {/* Hero Section */}
        <div className="text-center mb-20 max-w-3xl mx-auto space-y-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-xs font-medium text-white/70 backdrop-blur-md"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            System Operational
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-7xl font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-b from-white via-white/90 to-white/50"
          >
            One Account. <br />
            <span className="text-white/40">Infinite Possibilities.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg text-white/50 max-w-xl mx-auto leading-relaxed"
          >
            Zenchi unifies your digital existence. Navigate seamlessly between productivity, creativity, and entertainment in a single, immersive multiverse.
          </motion.p>
        </div>

        {/* Stats Row */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-20 w-full max-w-3xl"
        >
          {stats.map((stat) => (
            <div key={stat.id} className="flex flex-col items-center justify-center p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm hover:bg-white/10 transition-colors duration-300">
              <stat.icon className="text-white/40 mb-2" size={24} weight="duotone" />
              <div className="text-3xl font-bold tracking-tight text-white">
                <Counter value={stat.value} />
              </div>
              <div className="text-xs font-medium text-white/40 uppercase tracking-widest mt-1">
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>

        {/* Universe Selector */}
        <div className="w-full">
          <div className="flex items-center gap-4 mb-8">
            <h2 className="text-xl font-semibold text-white/90">Select Universe</h2>
            <div className="h-px bg-white/10 flex-1" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {universes.map((universe, index) => (
              <motion.button
                key={universe.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 + index * 0.1 }}
                onClick={() => router.push(universe.route)}
                onMouseEnter={() => setHoveredUniverse(universe.id)}
                onMouseLeave={() => setHoveredUniverse(null)}
                className={cn(
                  "group relative h-80 rounded-3xl p-6 text-left flex flex-col justify-between overflow-hidden transition-all duration-500",
                  "border border-white/10 bg-black/40 hover:shadow-2xl hover:-translate-y-1",
                  universe.borderHover
                )}
              >
                {/* Background Gradient on Hover */}
                <div className={cn(
                  "absolute inset-0 bg-gradient-to-b opacity-0 group-hover:opacity-100 transition-opacity duration-500 ease-out",
                  universe.gradient
                )} />

                {/* Content */}
                <div className="relative z-10">
                  <div className={cn(
                    "w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-all duration-300",
                    "bg-white/5 border border-white/10 text-white/70",
                    "group-hover:scale-110",
                    universe.textHover
                  )}>
                    <universe.icon size={28} weight="fill" />
                  </div>

                  <div className="space-y-1">
                    <div className="text-xs font-medium text-white/50 uppercase tracking-wider">{universe.tagline}</div>
                    <h3 className="text-2xl font-bold text-white group-hover:text-white transition-colors">{universe.name}</h3>
                  </div>

                  <p className="mt-4 text-sm text-white/40 leading-relaxed group-hover:text-white/60 transition-colors">
                    {universe.description}
                  </p>
                </div>

                {/* Footer Action */}
                <div className="relative z-10 flex items-center justify-between pt-6 border-t border-white/5 group-hover:border-white/10 transition-colors">
                  <span className="text-xs font-medium text-white/40 group-hover:text-white/90 transition-colors">
                    Enter Portal
                  </span>
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center border border-white/10 bg-white/5 transition-all duration-300",
                    "group-hover:bg-white group-hover:text-black group-hover:border-transparent group-hover:rotate-[-45deg]"
                  )}>
                    <ArrowRight size={14} weight="bold" />
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        </div>

      </main>

      {/* Footer */}
      <footer className="relative z-10 w-full py-8 text-center text-white/20 text-xs">
        <div className="max-w-7xl mx-auto px-6 border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p>© 2026 Zenchi Project. Evolution Protocol v4.0</p>
          <div className="flex gap-6">
            <span className="hover:text-white/40 cursor-pointer transition-colors">Privacy</span>
            <span className="hover:text-white/40 cursor-pointer transition-colors">Terms</span>
            <span className="hover:text-white/40 cursor-pointer transition-colors">Status</span>
          </div>
        </div>
      </footer>

    </div>
  )
}
