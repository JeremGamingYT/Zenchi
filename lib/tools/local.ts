import { tool } from "ai"
import { z } from "zod"
import fs from "fs/promises"
import path from "path"
import { exec } from "child_process"
import { promisify } from "util"

const execAsync = promisify(exec)

export const localTools = {
    list_files: tool({
        description: "List files and directories at a specific path on the local user's PC.",
        parameters: z.object({
            path: z.string().describe("The absolute or relative path to list files from. Defaults to current directory if empty."),
        }),
        execute: async ({ path: dirPath }: { path?: string }) => {
            try {
                const targetPath = dirPath ? path.resolve(process.cwd(), dirPath) : process.cwd()
                const entries = await fs.readdir(targetPath, { withFileTypes: true })

                const files = entries.map(entry => ({
                    name: entry.name,
                    type: entry.isDirectory() ? "directory" : "file",
                }))

                return {
                    path: targetPath,
                    files,
                }
            } catch (error: any) {
                return { error: `Failed to list files: ${error.message}` }
            }
        },
    }),

    read_file: tool({
        description: "Read the content of a specific file on the local user's PC.",
        parameters: z.object({
            path: z.string().describe("The path of the file to read."),
        }),
        execute: async ({ path: filePath }: { path: string }) => {
            try {
                const targetPath = path.resolve(process.cwd(), filePath)
                const content = await fs.readFile(targetPath, "utf-8")
                return {
                    path: targetPath,
                    content,
                }
            } catch (error: any) {
                return { error: `Failed to read file: ${error.message}` }
            }
        },
    }),

    execute_command: tool({
        description: "Execute a shell command on the local user's PC. Use with caution.",
        parameters: z.object({
            command: z.string().describe("The shell command to execute."),
        }),
        execute: async ({ command }: { command: string }) => {
            try {
                console.log(`[Tool] Executing command: ${command}`)
                const { stdout, stderr } = await execAsync(command)
                return {
                    stdout,
                    stderr,
                }
            } catch (error: any) {
                return {
                    error: `Command failed: ${error.message}`,
                    stdout: error.stdout,
                    stderr: error.stderr
                }
            }
        },
    }),
}
