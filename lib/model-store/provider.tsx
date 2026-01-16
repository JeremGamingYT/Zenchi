"use client"

import { fetchClient } from "@/lib/fetch"
import { ModelConfig } from "@/lib/models/types"
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react"

type UserKeyStatus = {
  openrouter: boolean
  openai: boolean
  mistral: boolean
  google: boolean
  perplexity: boolean
  xai: boolean
  anthropic: boolean
  [key: string]: boolean // Allow for additional providers
}

type ModelContextType = {
  models: ModelConfig[]
  userKeyStatus: UserKeyStatus
  favoriteModels: string[]
  isLoading: boolean
  refreshModels: () => Promise<void>
  refreshUserKeyStatus: () => Promise<void>
  refreshFavoriteModels: () => Promise<void>
  refreshFavoriteModelsSilent: () => Promise<void>
  refreshAll: () => Promise<void>
}

const ModelContext = createContext<ModelContextType | undefined>(undefined)

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [models, setModels] = useState<ModelConfig[]>([])
  const [userKeyStatus, setUserKeyStatus] = useState<UserKeyStatus>({
    openrouter: false,
    openai: false,
    mistral: false,
    google: false,
    perplexity: false,
    xai: false,
    anthropic: false,
  })
  const [favoriteModels, setFavoriteModels] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)

  const OLLAMA_STORAGE_KEY = "zenchi-ollama-settings"

  // Get Ollama endpoint from localStorage
  const getOllamaEndpoint = useCallback((): string | null => {
    if (typeof window === "undefined") return null
    try {
      const stored = localStorage.getItem(OLLAMA_STORAGE_KEY)
      if (stored) {
        const settings = JSON.parse(stored)
        if (settings.enabled) {
          return settings.connectionType === "local"
            ? settings.localEndpoint
            : settings.remoteEndpoint
        }
      }
    } catch {
      // Ignore parsing errors
    }
    return null
  }, [])

  const fetchModels = useCallback(async () => {
    try {
      // Fetch standard models
      const response = await fetchClient("/api/models")
      let allModels: ModelConfig[] = []

      if (response.ok) {
        const data = await response.json()
        allModels = data.models || []
      }

      // Fetch Ollama models if endpoint is configured
      const ollamaEndpoint = getOllamaEndpoint()
      if (ollamaEndpoint) {
        try {
          const ollamaResponse = await fetch("/api/ollama/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ endpoint: ollamaEndpoint }),
          })

          if (ollamaResponse.ok) {
            const ollamaData = await ollamaResponse.json()
            const ollamaModels = ollamaData.models || []
            // Merge Ollama models at the beginning of the list
            allModels = [...ollamaModels, ...allModels]
          }
        } catch (error) {
          console.warn("Failed to fetch Ollama models:", error)
        }
      }

      setModels(allModels)
    } catch (error) {
      console.error("Failed to fetch models:", error)
    }
  }, [getOllamaEndpoint])

  const fetchUserKeyStatus = useCallback(async () => {
    try {
      const response = await fetchClient("/api/user-key-status")
      if (response.ok) {
        const data = await response.json()
        setUserKeyStatus(data)
      }
    } catch (error) {
      console.error("Failed to fetch user key status:", error)
      // Set default values on error
      setUserKeyStatus({
        openrouter: false,
        openai: false,
        mistral: false,
        google: false,
        perplexity: false,
        xai: false,
        anthropic: false,
      })
    }
  }, [])

  const fetchFavoriteModels = useCallback(async () => {
    try {
      const response = await fetchClient(
        "/api/user-preferences/favorite-models"
      )
      if (response.ok) {
        const data = await response.json()
        setFavoriteModels(data.favorite_models || [])
      }
    } catch (error) {
      console.error("Failed to fetch favorite models:", error)
      setFavoriteModels([])
    }
  }, [])

  const refreshModels = useCallback(async () => {
    setIsLoading(true)
    try {
      await fetchModels()
    } finally {
      setIsLoading(false)
    }
  }, [fetchModels])

  const refreshUserKeyStatus = useCallback(async () => {
    setIsLoading(true)
    try {
      await fetchUserKeyStatus()
    } finally {
      setIsLoading(false)
    }
  }, [fetchUserKeyStatus])

  const refreshFavoriteModels = useCallback(async () => {
    setIsLoading(true)
    try {
      await fetchFavoriteModels()
    } finally {
      setIsLoading(false)
    }
  }, [fetchFavoriteModels])

  const refreshFavoriteModelsSilent = useCallback(async () => {
    try {
      await fetchFavoriteModels()
    } catch (error) {
      console.error(
        "❌ ModelProvider: Failed to silently refresh favorite models:",
        error
      )
    }
  }, [fetchFavoriteModels])

  const refreshAll = useCallback(async () => {
    setIsLoading(true)
    try {
      await Promise.all([
        fetchModels(),
        fetchUserKeyStatus(),
        fetchFavoriteModels(),
      ])
    } finally {
      setIsLoading(false)
    }
  }, [fetchModels, fetchUserKeyStatus, fetchFavoriteModels])

  // Initial data fetch
  useEffect(() => {
    refreshAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount

  return (
    <ModelContext.Provider
      value={{
        models,
        userKeyStatus,
        favoriteModels,
        isLoading,
        refreshModels,
        refreshUserKeyStatus,
        refreshFavoriteModels,
        refreshFavoriteModelsSilent,
        refreshAll,
      }}
    >
      {children}
    </ModelContext.Provider>
  )
}

// Custom hook to use the model context
export function useModel() {
  const context = useContext(ModelContext)
  if (context === undefined) {
    throw new Error("useModel must be used within a ModelProvider")
  }
  return context
}
