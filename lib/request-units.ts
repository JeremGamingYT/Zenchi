export const FREE_PLAN_DAILY_REQUEST_UNITS = 10
export const PRO_PLAN_DAILY_REQUEST_UNITS = 500

export const REQUEST_MULTIPLIER_TIERS = [
  { maxCostPerMillion: 2, multiplier: 1 },
  { maxCostPerMillion: 8, multiplier: 2 },
  { maxCostPerMillion: 20, multiplier: 3 },
  { maxCostPerMillion: 40, multiplier: 4 },
  { maxCostPerMillion: 80, multiplier: 5 },
] as const

export function getRequestMultiplierFromPricing(params: {
  inputCostPerMillion?: number
  outputCostPerMillion?: number
}): number {
  const input = params.inputCostPerMillion ?? 0
  const output = params.outputCostPerMillion ?? 0

  if (input <= 0 && output <= 0) {
    return 1
  }

  const weightedCost = (input * 0.35) + (output * 0.65)

  const tier = REQUEST_MULTIPLIER_TIERS.find(
    (entry) => weightedCost <= entry.maxCostPerMillion
  )

  return tier?.multiplier ?? 6
}
