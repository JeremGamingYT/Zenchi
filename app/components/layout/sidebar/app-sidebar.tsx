"use client"

import { groupChatsByDate } from "@/app/components/history/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ZolaIcon } from "@/components/icons/zola"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
} from "@/components/ui/sidebar"
import { useChats } from "@/lib/chat-store/chats/provider"
import { APP_NAME } from "@/lib/config"
import {
  ChatTeardropText,
  GithubLogo,
  MagnifyingGlass,
  NotePencilIcon,
} from "@phosphor-icons/react"
import { Pin } from "lucide-react"
import Link from "next/link"
import { useParams, useRouter } from "next/navigation"
import { useMemo } from "react"
import { HistoryTrigger } from "../../history/history-trigger"
import { HeaderSidebarTrigger } from "../header-sidebar-trigger"
import { SidebarList } from "./sidebar-list"
import { SidebarProject } from "./sidebar-project"

export function AppSidebar() {
  const { chats, pinnedChats, isLoading } = useChats()
  const params = useParams<{ chatId: string }>()
  const currentChatId = params.chatId

  const groupedChats = useMemo(() => {
    const result = groupChatsByDate(chats, "")
    return result
  }, [chats])
  const hasChats = chats.length > 0
  const router = useRouter()

  return (
    <Sidebar
      collapsible="offcanvas"
      variant="sidebar"
      className="bg-transparent"
    >
      <SidebarHeader className="h-14 px-3">
        <div className="flex items-center justify-between gap-2">
          <Link
            href="/chat"
            className="pointer-events-auto inline-flex items-center text-base font-medium tracking-tight"
          >
            <ZolaIcon className="mr-1 size-4" />
            {APP_NAME}
          </Link>
          <HeaderSidebarTrigger />
        </div>
      </SidebarHeader>
      <SidebarContent className="border-border/40 border-t">
        <ScrollArea className="flex h-full px-3 [&>div>div]:!block">
          <div className="mt-3 mb-5 flex w-full flex-col items-start gap-0">
            <button
              className="hover:bg-accent/80 hover:text-foreground text-primary group/new-chat relative inline-flex w-full items-center rounded-md bg-transparent px-2 py-2 text-sm transition-colors"
              type="button"
              onClick={() => router.push("/chat")}
            >
              <div className="flex items-center gap-2">
                <NotePencilIcon size={20} />
                New Chat
              </div>
              <div className="text-muted-foreground ml-auto text-xs opacity-0 duration-150 group-hover/new-chat:opacity-100">
                ⌘⇧U
              </div>
            </button>
            <HistoryTrigger
              hasSidebar={false}
              classNameTrigger="bg-transparent hover:bg-accent/80 hover:text-foreground text-primary relative inline-flex w-full items-center rounded-md px-2 py-2 text-sm transition-colors group/search"
              icon={<MagnifyingGlass size={24} className="mr-2" />}
              label={
                <div className="flex w-full items-center gap-2">
                  <span>Search</span>
                  <div className="text-muted-foreground ml-auto text-xs opacity-0 duration-150 group-hover/search:opacity-100">
                    ⌘+K
                  </div>
                </div>
              }
              hasPopover={false}
            />

          </div>
          <SidebarProject />
          {isLoading ? (
            <div className="h-full" />
          ) : hasChats ? (
            <div className="space-y-5">
              {pinnedChats.length > 0 && (
                <div className="space-y-5">
                  <SidebarList
                    key="pinned"
                    title="Pinned"
                    icon={<Pin className="size-3" />}
                    items={pinnedChats}
                    currentChatId={currentChatId}
                  />
                </div>
              )}
              {groupedChats?.map((group) => (
                <SidebarList
                  key={group.name}
                  title={group.name}
                  items={group.chats}
                  currentChatId={currentChatId}
                />
              ))}
            </div>
          ) : (
            <div className="flex h-[calc(100vh-160px)] flex-col items-center justify-center">
              <ChatTeardropText
                size={24}
                className="text-muted-foreground mb-1 opacity-40"
              />
              <div className="text-muted-foreground text-center">
                <p className="mb-1 text-base font-medium">No chats yet</p>
                <p className="text-sm opacity-70">Start a new conversation</p>
              </div>
            </div>
          )}
        </ScrollArea>
      </SidebarContent>
      <SidebarFooter className="border-border/40 mb-2 border-t p-3">
        <Link
          href="/pricing"
          className="hover:bg-sidebar-accent/80 border-sidebar-border/60 bg-sidebar-accent/40 flex items-center gap-2 rounded-xl border p-2.5 transition-colors"
          aria-label="Open pricing"
        >
          <div className="rounded-full border border-current/20 p-1">
            <GithubLogo className="size-4" />
          </div>
          <div className="flex flex-col">
            <div className="text-sidebar-foreground text-sm font-medium">
              Current plan
            </div>
            <div className="text-sidebar-foreground/70 text-xs">
              View subscription & limits
            </div>
          </div>
        </Link>
      </SidebarFooter>
    </Sidebar>
  )
}
