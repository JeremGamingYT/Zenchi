"use client"

import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Sparkle } from "@phosphor-icons/react"
import { useRouter } from "next/navigation"

interface FeatureUnavailableDialogProps {
    open: boolean
    onOpenChange: (open: boolean) => void
}

export function FeatureUnavailableDialog({
    open,
    onOpenChange,
}: FeatureUnavailableDialogProps) {
    const router = useRouter()

    const handleGoToKoma = () => {
        onOpenChange(false)
        router.push("/koma")
    }

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-md">
                <DialogHeader>
                    <DialogTitle className="text-xl">
                        Feature Not Available Yet
                    </DialogTitle>
                    <DialogDescription className="pt-2 text-base">
                        This feature is not yet available! Head to the sidebar and click on{" "}
                        <span className="font-semibold text-foreground">&apos;Koma&apos;</span>{" "}
                        to use our most powerful tools for anime/webtoon and manga creators
                        (coming soon)!
                    </DialogDescription>
                </DialogHeader>
                <DialogFooter className="mt-6 flex gap-2 sm:justify-end">
                    <Button
                        variant="secondary"
                        onClick={() => onOpenChange(false)}
                    >
                        Close
                    </Button>
                    <Button
                        onClick={handleGoToKoma}
                    >
                        <Sparkle className="mr-2 size-4" weight="fill" />
                        Go to Koma
                    </Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    )
}
