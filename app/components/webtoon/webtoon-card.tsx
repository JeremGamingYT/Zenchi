"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"
import { Eye, Heart, Star } from "@phosphor-icons/react"
import Link from "next/link"

interface WebtoonCardProps {
    id: string
    title: string
    author: string
    cover: string
    genre: string
    views: string
    likes: string
    rating: number
    isNew?: boolean
    isHot?: boolean
    className?: string
}

export function WebtoonCard({
    id,
    title,
    author,
    cover,
    genre,
    views,
    likes,
    rating,
    isNew,
    isHot,
    className
}: WebtoonCardProps) {
    return (
        <Link href={`/webtoon/series/${id}`}>
            <motion.div
                whileHover={{ y: -8, scale: 1.02 }}
                transition={{ type: "spring", bounce: 0.3 }}
                className={cn(
                    "group relative rounded-2xl overflow-hidden bg-white/5 border border-white/10 cursor-pointer",
                    "hover:border-emerald-500/30 hover:shadow-xl hover:shadow-emerald-500/10",
                    "transition-colors duration-300",
                    className
                )}
            >
                {/* Cover Image */}
                <div className="relative aspect-[3/4] overflow-hidden">
                    <img
                        src={cover}
                        alt={title}
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    />

                    {/* Gradient Overlay */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />

                    {/* Badges */}
                    <div className="absolute top-3 left-3 flex gap-2">
                        {isNew && (
                            <span className="px-2 py-0.5 rounded-md bg-emerald-500 text-[10px] font-bold uppercase tracking-wider">
                                New
                            </span>
                        )}
                        {isHot && (
                            <span className="px-2 py-0.5 rounded-md bg-orange-500 text-[10px] font-bold uppercase tracking-wider">
                                Hot
                            </span>
                        )}
                    </div>

                    {/* Rating */}
                    <div className="absolute top-3 right-3 flex items-center gap-1 px-2 py-1 rounded-lg bg-black/50 backdrop-blur-sm">
                        <Star size={12} weight="fill" className="text-amber-400" />
                        <span className="text-xs font-semibold">{rating.toFixed(1)}</span>
                    </div>

                    {/* Genre Tag */}
                    <div className="absolute bottom-3 left-3">
                        <span className="px-2.5 py-1 rounded-full bg-white/10 backdrop-blur-sm text-[11px] font-medium text-white/80">
                            {genre}
                        </span>
                    </div>
                </div>

                {/* Content */}
                <div className="p-4 space-y-2">
                    <h3 className="font-bold text-white group-hover:text-emerald-400 transition-colors line-clamp-1">
                        {title}
                    </h3>
                    <p className="text-sm text-white/50">{author}</p>

                    {/* Stats */}
                    <div className="flex items-center gap-4 pt-2">
                        <div className="flex items-center gap-1.5 text-white/40">
                            <Eye size={14} />
                            <span className="text-xs">{views}</span>
                        </div>
                        <div className="flex items-center gap-1.5 text-white/40">
                            <Heart size={14} />
                            <span className="text-xs">{likes}</span>
                        </div>
                    </div>
                </div>
            </motion.div>
        </Link>
    )
}
