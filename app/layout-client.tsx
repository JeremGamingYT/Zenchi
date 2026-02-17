"use client"

import { API_ROUTE_CSRF } from "@/lib/routes"
import { useQuery } from "@tanstack/react-query"
import { useEffect } from "react"

export function LayoutClient() {
  useEffect(() => {
    if (process.env.NODE_ENV !== "development") {
      return
    }

    if (!("serviceWorker" in navigator)) {
      return
    }

    navigator.serviceWorker.getRegistrations().then((registrations) => {
      registrations.forEach((registration) => {
        registration.unregister()
      })
    })
  }, [])

  useQuery({
    queryKey: ["csrf-init"],
    queryFn: async () => {
      await fetch(API_ROUTE_CSRF)
      return true
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    retry: false,
  })

  return null
}
