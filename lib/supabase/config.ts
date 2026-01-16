function isValidSupabaseConfig(): boolean {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  // Check if values exist and are not placeholders
  if (!url || !key) return false
  if (url.includes("your_") || key.includes("your_")) return false

  // Validate URL format
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}

export const isSupabaseEnabled = isValidSupabaseConfig()
