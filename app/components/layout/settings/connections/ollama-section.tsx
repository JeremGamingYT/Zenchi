"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { toast } from "@/components/ui/toast"
import { useState, useEffect } from "react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { GlobeSimple, HardDrive, CheckCircle, XCircle, SpinnerGap } from "@phosphor-icons/react"

const OLLAMA_STORAGE_KEY = "zenchi-ollama-settings"

type ConnectionType = "local" | "remote"

interface OllamaSettings {
  connectionType: ConnectionType
  localEndpoint: string
  remoteEndpoint: string
  enabled: boolean
}

const defaultSettings: OllamaSettings = {
  connectionType: "local",
  localEndpoint: "http://localhost:11434",
  remoteEndpoint: "",
  enabled: true,
}

function getStoredSettings(): OllamaSettings {
  if (typeof window === "undefined") return defaultSettings
  try {
    const stored = localStorage.getItem(OLLAMA_STORAGE_KEY)
    if (stored) return JSON.parse(stored)
  } catch {
    // Ignore
  }
  return defaultSettings
}

function saveSettings(settings: OllamaSettings) {
  if (typeof window === "undefined") return
  localStorage.setItem(OLLAMA_STORAGE_KEY, JSON.stringify(settings))
}

export function OllamaSection() {
  const [settings, setSettings] = useState<OllamaSettings>(defaultSettings)
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "success" | "error">("idle")

  useEffect(() => {
    setSettings(getStoredSettings())
  }, [])

  const updateSettings = (update: Partial<OllamaSettings>) => {
    const newSettings = { ...settings, ...update }
    setSettings(newSettings)
    saveSettings(newSettings)
    setConnectionStatus("idle")
  }

  const currentEndpoint = settings.connectionType === "local"
    ? settings.localEndpoint
    : settings.remoteEndpoint

  const testConnection = async () => {
    if (!currentEndpoint) {
      toast({
        title: "No endpoint configured",
        description: "Please enter an endpoint URL first.",
        status: "error",
      })
      return
    }

    setIsLoading(true)
    setConnectionStatus("idle")

    try {
      let data: { models?: { name: string }[] }

      if (settings.connectionType === "remote") {
        // Use server-side proxy for remote connections to bypass CORS
        const proxyResponse = await fetch("/api/ollama-proxy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            endpoint: currentEndpoint,
            path: "/api/tags"
          }),
        })

        if (!proxyResponse.ok) {
          const errorData = await proxyResponse.json().catch(() => ({}))
          throw new Error(errorData.error || `Server returned ${proxyResponse.status}`)
        }

        data = await proxyResponse.json()
      } else {
        // Direct connection for local Ollama
        const response = await fetch(`${currentEndpoint}/api/tags`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        })

        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${response.statusText}`)
        }

        data = await response.json()
      }

      const modelCount = data.models?.length || 0
      setConnectionStatus("success")
      toast({
        title: "Connection successful!",
        description: `Found ${modelCount} model${modelCount !== 1 ? "s" : ""} available.`,
      })
    } catch (error) {
      setConnectionStatus("error")
      const errorMsg = error instanceof Error ? error.message : "Unknown error"
      toast({
        title: "Connection failed",
        description: settings.connectionType === "remote"
          ? `Remote connection error: ${errorMsg}`
          : "Ensure Ollama is running locally (ollama serve).",
        status: "error",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-lg font-medium">Ollama Connection</h3>
        <p className="text-muted-foreground text-sm">
          Connect to Ollama for running open-source LLMs locally or via a remote server (NGrok).
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {settings.connectionType === "local" ? (
                <HardDrive className="size-5" />
              ) : (
                <GlobeSimple className="size-5" />
              )}
              <span>Ollama</span>
              {connectionStatus === "success" && <CheckCircle className="size-4 text-green-500" weight="fill" />}
              {connectionStatus === "error" && <XCircle className="size-4 text-red-500" weight="fill" />}
            </div>
            <Switch
              checked={settings.enabled}
              onCheckedChange={(enabled) => updateSettings({ enabled })}
            />
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Connection Type Selector */}
          <div className="space-y-2">
            <Label>Connection Type</Label>
            <Select
              value={settings.connectionType}
              onValueChange={(value: ConnectionType) => updateSettings({ connectionType: value })}
              disabled={!settings.enabled}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select connection type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="local">
                  <div className="flex items-center gap-2">
                    <HardDrive className="size-4" />
                    <span>Local (localhost)</span>
                  </div>
                </SelectItem>
                <SelectItem value="remote">
                  <div className="flex items-center gap-2">
                    <GlobeSimple className="size-4" />
                    <span>Remote (NGrok)</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Local Endpoint */}
          {settings.connectionType === "local" && (
            <div className="space-y-2">
              <Label htmlFor="local-endpoint">Local Endpoint</Label>
              <Input
                id="local-endpoint"
                type="url"
                placeholder="http://localhost:11434"
                value={settings.localEndpoint}
                onChange={(e) => updateSettings({ localEndpoint: e.target.value })}
                disabled={!settings.enabled}
              />
              <p className="text-muted-foreground text-xs">
                Default Ollama endpoint. Run <code className="bg-muted px-1 rounded">ollama serve</code> to start.
              </p>
            </div>
          )}

          {/* Remote/NGrok Endpoint */}
          {settings.connectionType === "remote" && (
            <div className="space-y-2">
              <Label htmlFor="remote-endpoint">NGrok URL</Label>
              <Input
                id="remote-endpoint"
                type="url"
                placeholder="https://xxxx-xxx-xxx.ngrok-free.app"
                value={settings.remoteEndpoint}
                onChange={(e) => updateSettings({ remoteEndpoint: e.target.value })}
                disabled={!settings.enabled}
              />
              <p className="text-muted-foreground text-xs">
                Your NGrok tunnel URL. On remote server, run:{" "}
                <code className="bg-muted px-1 rounded">OLLAMA_HOST=0.0.0.0 ollama serve</code>{" "}
                then <code className="bg-muted px-1 rounded">ngrok http 11434</code>
              </p>
            </div>
          )}

          {/* Test Connection Button */}
          {settings.enabled && (
            <div className="flex gap-2 pt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={testConnection}
                disabled={isLoading || !currentEndpoint}
                className="gap-2"
              >
                {isLoading ? (
                  <>
                    <SpinnerGap className="size-4 animate-spin" />
                    Testing...
                  </>
                ) : (
                  "Test Connection"
                )}
              </Button>
            </div>
          )}

          {/* Instructions for Remote Setup */}
          {settings.connectionType === "remote" && settings.enabled && (
            <div className="rounded-md bg-blue-50 p-3 dark:bg-blue-950/20">
              <p className="text-sm text-blue-800 dark:text-blue-200 font-medium mb-2">
                Remote Setup Instructions:
              </p>
              <ol className="text-xs text-blue-700 dark:text-blue-300 space-y-1 list-decimal pl-4">
                <li>Install Ollama on your remote server</li>
                <li>Set environment: <code>OLLAMA_HOST=0.0.0.0</code></li>
                <li>Start Ollama: <code>ollama serve</code></li>
                <li>Install and authenticate NGrok</li>
                <li>Create tunnel: <code>ngrok http 11434</code></li>
                <li>Copy the NGrok URL above</li>
              </ol>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
