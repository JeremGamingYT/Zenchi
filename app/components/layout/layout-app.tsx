"use client"

import { Header } from "@/app/components/layout/header"
import { AppSidebar } from "@/app/components/layout/sidebar/app-sidebar"
import { BudgetWarnings } from "@/app/components/budget/budget-warnings"
import { useUserPreferences } from "@/lib/user-preference-store/provider"
import { useBudget } from "@/lib/budget-store/provider"
import { motion } from "framer-motion"

export function LayoutApp({ children }: { children: React.ReactNode }) {
  const { preferences } = useUserPreferences()
  const { budgetLimits } = useBudget()
  const hasSidebar = preferences.layout === "sidebar"
  const showBudgetWarnings =
    Array.isArray(budgetLimits) &&
    budgetLimits.length > 0 &&
    budgetLimits.some((b) => b.in_app_notifications)

  return (
    <div className="bg-background relative flex h-dvh w-full overflow-hidden p-2">

      {/* Dynamic Violet Gradient Background */}
      <motion.div className="pointer-events-none fixed inset-[-5%] z-0 opacity-[0.18]">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(139,92,246,0.08),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(59,130,246,0.05),transparent_50%)]" />
      </motion.div>

      {hasSidebar && <AppSidebar />}
      <main className="@container bg-background/92 border-border/60 relative z-10 h-[calc(100dvh-1rem)] w-0 flex-shrink flex-grow overflow-y-auto rounded-2xl border shadow-sm backdrop-blur-sm">
        <Header hasSidebar={hasSidebar} />
        {children}
      </main>
      {showBudgetWarnings && <BudgetWarnings />}
    </div>
  )
}
