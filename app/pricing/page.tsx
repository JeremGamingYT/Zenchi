"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  FREE_PLAN_DAILY_REQUEST_UNITS,
  REQUEST_MULTIPLIER_TIERS,
  getRequestMultiplierFromPricing,
} from "@/lib/request-units"
import { ArrowLeft, CheckCircle2 } from "lucide-react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useEffect, useMemo, useState } from "react"

type UsageResponse = {
  usage?: Array<{ created_at?: string | null }>
}

function formatEuro(value: number) {
  return new Intl.NumberFormat("fr-FR", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value)
}

export default function PricingPage() {
  const [acceptPersonalizedOffer, setAcceptPersonalizedOffer] = useState(false)
  const [detectedMonthlyUsage, setDetectedMonthlyUsage] = useState(300)

  useEffect(() => {
    let cancelled = false

    const loadUsage = async () => {
      try {
        const response = await fetch("/api/model-usage?limit=1000&offset=0", {
          method: "GET",
          cache: "no-store",
        })

        if (!response.ok) {
          return
        }

        const payload = (await response.json()) as UsageResponse
        const usageRows = Array.isArray(payload.usage) ? payload.usage : []
        const now = new Date()
        const currentMonth = now.getMonth()
        const currentYear = now.getFullYear()

        const monthCount = usageRows.filter((row) => {
          if (!row.created_at) {
            return false
          }

          const date = new Date(row.created_at)
          return (
            date.getMonth() === currentMonth && date.getFullYear() === currentYear
          )
        }).length

        if (!cancelled && monthCount > 0) {
          setDetectedMonthlyUsage(monthCount)
        }
      } catch {
        // Keep fallback usage value
      }
    }

    loadUsage()

    return () => {
      cancelled = true
    }
  }, [])

  const proOffer = useMemo(() => {
    const normalizedUsage = Math.max(30, detectedMonthlyUsage)
    const baseMonthlyUnits = acceptPersonalizedOffer
      ? Math.max(200, Math.ceil(normalizedUsage * 1.25))
      : 600

    const monthlyPrice = acceptPersonalizedOffer
      ? Math.max(7, Number((baseMonthlyUnits * 0.022).toFixed(2)))
      : 19

    return {
      monthlyUnits: baseMonthlyUnits,
      monthlyPrice,
    }
  }, [acceptPersonalizedOffer, detectedMonthlyUsage])

  const hero = {
    initial: { opacity: 0, y: 14 },
    animate: { opacity: 1, y: 0 },
  }

  return (
    <main className="bg-background text-foreground relative min-h-dvh overflow-hidden">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="pointer-events-none absolute inset-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_-10%,oklch(0.72_0.09_280_/_0.18),transparent_55%)]" />
      </motion.div>

      <div className="relative mx-auto flex min-h-dvh w-full max-w-6xl flex-col px-4 py-6 sm:px-8 sm:py-8">
        <motion.div {...hero} transition={{ duration: 0.35 }} className="mb-6 flex items-center justify-between">
          <Button asChild variant="ghost" className="rounded-xl">
            <Link href="/chat" className="inline-flex items-center gap-2">
              <ArrowLeft className="size-4" />
              Back to chat
            </Link>
          </Button>
          <div className="text-muted-foreground text-xs sm:text-sm">Transparent request units billing</div>
        </motion.div>

        <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col items-center justify-center gap-8">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.05 }}
            className="space-y-3 text-center"
          >
            <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">Pricing</h1>
            <p className="text-muted-foreground mx-auto max-w-2xl text-sm sm:text-base">
              Tous les modèles sont accessibles. Les modèles plus coûteux consomment plus d’unités via un multiplicateur automatique.
            </p>
          </motion.div>

          <motion.section
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.1 }}
            className="grid w-full items-stretch gap-5 lg:grid-cols-3"
          >
            <Card className="bg-card/85 border-border/70 rounded-2xl">
              <CardHeader>
                <CardTitle>Free</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p className="text-foreground text-2xl font-semibold">{formatEuro(0)} / month</p>
                <p>{FREE_PLAN_DAILY_REQUEST_UNITS} request units / day</p>
                <p>API keys required</p>
                <p>Access to all models</p>
              </CardContent>
            </Card>

            <Card className="bg-card border-primary/30 relative rounded-2xl border-2 shadow-xl">
              <div className="bg-primary text-primary-foreground absolute top-3 right-3 rounded-full px-2.5 py-1 text-xs font-medium">
                Pro
              </div>
              <CardHeader className="pb-2 text-center">
                <CardTitle>Pro</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="space-y-1 text-center">
                  <p className="text-foreground text-4xl font-semibold">{formatEuro(proOffer.monthlyPrice)}</p>
                  <p className="text-muted-foreground text-sm">{proOffer.monthlyUnits.toLocaleString("fr-FR")} request units / month</p>
                </div>

                <div className="bg-muted/45 border-border/60 rounded-xl border p-3">
                  <div className="text-sm font-medium">Offre personnalisée (optionnel)</div>
                  <p className="text-muted-foreground mt-1 text-xs">
                    Usage détecté: {detectedMonthlyUsage} requêtes ce mois. Si activé, le plan s’ajuste automatiquement à votre profil.
                  </p>
                  <label className="mt-3 flex items-start gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={acceptPersonalizedOffer}
                      onChange={(event) => setAcceptPersonalizedOffer(event.target.checked)}
                      className="mt-0.5"
                    />
                    J’accepte l’ajustement automatique de mon plan.
                  </label>
                </div>

                <Button className="w-full rounded-xl">Choose Pro</Button>
              </CardContent>
            </Card>

            <Card className="bg-card/85 border-border/70 rounded-2xl">
              <CardHeader>
                <CardTitle>Enterprise</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p className="text-foreground text-2xl font-semibold">Custom</p>
                <p>Custom unit pool</p>
                <p>Advanced governance</p>
                <p>Priority SLA support</p>
              </CardContent>
            </Card>
          </motion.section>

          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.14 }}
            className="bg-card/75 border-border/60 w-full rounded-2xl border p-6"
          >
            <h2 className="text-center text-lg font-semibold">Model multipliers</h2>
            <p className="text-muted-foreground mx-auto mt-2 max-w-3xl text-center text-sm">
              Le multiplicateur est calculé sur le coût input/output du modèle. Plus le modèle est cher, plus une requête consomme d’unités.
            </p>

            <div className="mt-5 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
              {REQUEST_MULTIPLIER_TIERS.map((tier) => (
                <div key={tier.maxCostPerMillion} className="bg-muted/35 border-border/60 rounded-xl border p-3">
                  <div className="font-medium">Up to ${tier.maxCostPerMillion}/1M tokens</div>
                  <div className="text-muted-foreground mt-1">Multiplier x{tier.multiplier}</div>
                </div>
              ))}
              <div className="bg-muted/35 border-border/60 rounded-xl border p-3">
                <div className="font-medium">Above last tier</div>
                <div className="text-muted-foreground mt-1">Multiplier x6</div>
              </div>
            </div>

            <div className="mt-5 grid gap-3 text-sm sm:grid-cols-2">
              {[
                `Example A: model at $1.5/$6.0 => x${getRequestMultiplierFromPricing({ inputCostPerMillion: 1.5, outputCostPerMillion: 6.0 })}`,
                `Example B: model at $15/$60 => x${getRequestMultiplierFromPricing({ inputCostPerMillion: 15, outputCostPerMillion: 60 })}`,
              ].map((item) => (
                <div key={item} className="bg-muted/35 border-border/60 flex items-start gap-2 rounded-xl border p-3">
                  <CheckCircle2 className="text-primary mt-0.5 size-4" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </motion.section>
        </div>
      </div>
    </main>
  )
}
