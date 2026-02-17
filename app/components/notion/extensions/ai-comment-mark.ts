import { Mark, mergeAttributes } from '@tiptap/core'

export interface AICommentOptions {
    HTMLAttributes: Record<string, unknown>
}

declare module '@tiptap/core' {
    interface Commands<ReturnType> {
        aiComment: {
            setAIComment: (comment: string) => ReturnType
            unsetAIComment: () => ReturnType
        }
    }
}

export const AIComment = Mark.create<AICommentOptions>({
    name: 'aiComment',

    addOptions() {
        return {
            HTMLAttributes: {},
        }
    },

    addAttributes() {
        return {
            comment: {
                default: null,
                parseHTML: element => element.getAttribute('data-comment'),
                renderHTML: attributes => {
                    if (!attributes.comment) {
                        return {}
                    }
                    return {
                        'data-comment': attributes.comment,
                    }
                },
            },
            id: {
                default: null,
                parseHTML: element => element.getAttribute('data-comment-id'),
                renderHTML: attributes => {
                    return {
                        'data-comment-id': attributes.id || `comment-${Date.now()}`,
                    }
                },
            },
        }
    },

    parseHTML() {
        return [
            {
                tag: 'span[data-comment]',
            },
        ]
    },

    renderHTML({ HTMLAttributes }) {
        return [
            'span',
            mergeAttributes(this.options.HTMLAttributes, HTMLAttributes, {
                class: 'ai-comment-mark',
                style: 'background: rgba(139, 92, 246, 0.15); border-bottom: 2px solid rgba(139, 92, 246, 0.5); cursor: pointer; padding: 0 2px; border-radius: 2px;',
            }),
            0,
        ]
    },

    addCommands() {
        return {
            setAIComment:
                (comment: string) =>
                    ({ commands }) => {
                        return commands.setMark(this.name, { comment, id: `comment-${Date.now()}` })
                    },
            unsetAIComment:
                () =>
                    ({ commands }) => {
                        return commands.unsetMark(this.name)
                    },
        }
    },
})
