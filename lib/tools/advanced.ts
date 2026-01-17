import { tool } from "ai"
import { z } from "zod"
import fs from "fs/promises"
import path from "path"
import { exec } from "child_process"
import { promisify } from "util"

const execAsync = promisify(exec)

const editFileSchema = z.object({
    path: z.string().describe("The absolute path to the file to edit."),
    target: z.string().describe("The exact string in the file to be replaced."),
    replacement: z.string().describe("The new string to replace the target with."),
})

const createDirSchema = z.object({
    path: z.string().describe("The path to the new directory."),
})

const deleteFileSchema = z.object({
    path: z.string().describe("The path to the file to delete."),
    confirm: z.boolean().optional().describe("Set to true to confirm deletion. If false or missing, the tool will fail."),
})

const deleteDirSchema = z.object({
    path: z.string().describe("The path to the directory to delete."),
    confirm: z.boolean().optional().describe("Set to true to confirm deletion. If false or missing, the tool will fail."),
})

const runCommandSchema = z.object({
    command: z.string().describe("The command to execute."),
    confirm: z.boolean().optional().describe("Set to true to confirm execution. If false or missing, the tool will fail."),
})

export const advancedTools = {
    edit_file: tool({
        description: "Edit a file by replacing a specific target string with a replacement string. Use this for precise code modifications.",
        parameters: editFileSchema,
        execute: async (args: z.infer<typeof editFileSchema>) => {
            const { path: filePath, target, replacement } = args
            try {
                const targetPath = path.resolve(process.cwd(), filePath)

                const content = await fs.readFile(targetPath, "utf-8")

                if (!content.includes(target)) {
                    return { error: "Target string not found in file. Please ensure exact match including whitespace." }
                }

                const newContent = content.replace(target, replacement)

                await fs.writeFile(targetPath, newContent, "utf-8")

                return { success: true, message: `Successfully edited ${filePath}` }
            } catch (error: any) {
                return { error: `Failed to edit file: ${error.message}` }
            }
        },
    }),

    create_directory: tool({
        description: "Create a new directory.",
        parameters: createDirSchema,
        execute: async (args: z.infer<typeof createDirSchema>) => {
            const { path: dirPath } = args
            try {
                const targetPath = path.resolve(process.cwd(), dirPath)
                await fs.mkdir(targetPath, { recursive: true })
                return { success: true, message: `Created directory ${dirPath}` }
            } catch (error: any) {
                return { error: `Failed to create directory: ${error.message}` }
            }
        },
    }),

    delete_file: tool({
        description: "Delete a file. REQUIRES CONFIRMATION: You must ask the user for permission first. If approved, call this tool again with confirm: true.",
        parameters: deleteFileSchema,
        execute: async (args: z.infer<typeof deleteFileSchema>) => {
            const { path: filePath, confirm } = args
            if (confirm !== true) {
                return {
                    error: "CONFIRMATION REQUIRED",
                    message: "Please ask the user: 'Are you sure you want to delete " + filePath + "?' If they say yes, call this tool again with confirm: true."
                }
            }
            try {
                const targetPath = path.resolve(process.cwd(), filePath)
                await fs.unlink(targetPath)
                return { success: true, message: `Deleted file ${filePath}` }
            } catch (error: any) {
                return { error: `Failed to delete file: ${error.message}` }
            }
        },
    }),

    delete_directory: tool({
        description: "Delete a directory and its contents. REQUIRES CONFIRMATION: You must ask the user for permission first. If approved, call this tool again with confirm: true.",
        parameters: deleteDirSchema,
        execute: async (args: z.infer<typeof deleteDirSchema>) => {
            const { path: dirPath, confirm } = args
            if (confirm !== true) {
                return {
                    error: "CONFIRMATION REQUIRED",
                    message: "Please ask the user: 'Are you sure you want to delete the directory " + dirPath + " and all its contents?' If they say yes, call this tool again with confirm: true."
                }
            }
            try {
                const targetPath = path.resolve(process.cwd(), dirPath)
                await fs.rm(targetPath, { recursive: true, force: true })
                return { success: true, message: `Deleted directory ${dirPath}` }
            } catch (error: any) {
                return { error: `Failed to delete directory: ${error.message}` }
            }
        },
    }),

    run_command: tool({
        description: "Execute a shell command. REQUIRES CONFIRMATION: You must ask the user for permission first. If approved, call this tool again with confirm: true.",
        parameters: runCommandSchema,
        execute: async (args: z.infer<typeof runCommandSchema>) => {
            const { command, confirm } = args
            if (confirm !== true) {
                return {
                    error: "CONFIRMATION REQUIRED",
                    message: "Please ask the user: 'Do you want me to run this command: " + command + "?' If they say yes, call this tool again with confirm: true."
                }
            }
            try {
                console.log(`[Tool] Executing command: ${command}`)
                const { stdout, stderr } = await execAsync(command)
                return { stdout, stderr }
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
