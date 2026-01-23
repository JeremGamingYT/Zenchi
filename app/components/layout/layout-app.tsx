"use client"

import { Header } from "@/app/components/layout/header"
import { AppSidebar } from "@/app/components/layout/sidebar/app-sidebar"
import { BudgetWarnings } from "@/app/components/budget/budget-warnings"
import { useUserPreferences } from "@/lib/user-preference-store/provider"
import { useBudget } from "@/lib/budget-store/provider"

export function LayoutApp({ children }: { children: React.ReactNode }) {
  const { preferences } = useUserPreferences()
  const { budgetLimits } = useBudget()
  const hasSidebar = preferences.layout === "sidebar"
  const showBudgetWarnings =
    Array.isArray(budgetLimits) &&
    budgetLimits.length > 0 &&
    budgetLimits.some((b) => b.in_app_notifications)

  return (
    <div className="bg-background relative flex h-dvh w-full overflow-hidden">
      {/* Global Grainy Background */}
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.03] mix-blend-overlay">
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] brightness-100 contrast-150" />
      </div>

      {hasSidebar && <AppSidebar />}
      <main className="@container relative z-10 h-dvh w-0 flex-shrink flex-grow overflow-y-auto">
        <Header hasSidebar={hasSidebar} />
        {children}
      </main>
      {showBudgetWarnings && <BudgetWarnings />}
    </div>
  )
}
