import { Mark, mergeAttributes } from '@tiptap/core'

export interface AISuggestionOptions {
    HTMLAttributes: Record<string, unknown>
}

declare module '@tiptap/core' {
    interface Commands<ReturnType> {
        aiSuggestion: {
            setAISuggestion: (originalText: string, suggestedText: string, type: 'grammar' | 'style' | 'improve') => ReturnType
            unsetAISuggestion: () => ReturnType
        }
    }
}

export const AISuggestion = Mark.create<AISuggestionOptions>({
    name: 'aiSuggestion',

    addOptions() {
        return {
            HTMLAttributes: {},
        }
    },

    addAttributes() {
        return {
            originalText: {
                default: null,
                parseHTML: element => element.getAttribute('data-original'),
                renderHTML: attributes => ({
                    'data-original': attributes.originalText,
                }),
            },
            suggestedText: {
                default: null,
                parseHTML: element => element.getAttribute('data-suggested'),
                renderHTML: attributes => ({
                    'data-suggested': attributes.suggestedText,
                }),
            },
            suggestionType: {
                default: 'improve',
                parseHTML: element => element.getAttribute('data-type'),
                renderHTML: attributes => ({
                    'data-type': attributes.suggestionType,
                }),
            },
            id: {
                default: null,
                parseHTML: element => element.getAttribute('data-suggestion-id'),
                renderHTML: attributes => ({
                    'data-suggestion-id': attributes.id || `suggestion-${Date.now()}`,
                }),
            },
        }
    },

    parseHTML() {
        return [
            {
                tag: 'span[data-suggested]',
            },
        ]
    },

    renderHTML({ HTMLAttributes }) {
        const type = HTMLAttributes['data-type'] as string || 'improve'

        // Different colors for different suggestion types
        const colors: Record<string, { bg: string; border: string }> = {
            grammar: { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgba(34, 197, 94, 0.6)' },
            style: { bg: 'rgba(59, 130, 246, 0.15)', border: 'rgba(59, 130, 246, 0.6)' },
            improve: { bg: 'rgba(251, 191, 36, 0.15)', border: 'rgba(251, 191, 36, 0.6)' },
        }

        const color = colors[type] || colors.improve

        return [
            'span',
            mergeAttributes(this.options.HTMLAttributes, HTMLAttributes, {
                class: 'ai-suggestion-mark',
                style: `background: ${color.bg}; border-bottom: 2px dashed ${color.border}; cursor: pointer; padding: 0 2px; border-radius: 2px; transition: all 0.2s ease;`,
            }),
            0,
        ]
    },

    addCommands() {
        return {
            setAISuggestion:
                (originalText: string, suggestedText: string, type: 'grammar' | 'style' | 'improve') =>
                    ({ commands }) => {
                        return commands.setMark(this.name, {
                            originalText,
                            suggestedText,
                            suggestionType: type,
                            id: `suggestion-${Date.now()}`,
                        })
                    },
            unsetAISuggestion:
                () =>
                    ({ commands }) => {
                        return commands.unsetMark(this.name)
                    },
        }
    },
})
