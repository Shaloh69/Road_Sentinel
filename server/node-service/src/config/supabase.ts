import { logger } from "./logger";

// Supabase has been replaced by PC local storage served via the AI service
// (Cloudflare tunnel). See storage.service.ts and ai-service/app/main.py.
//
// This stub keeps server.ts unchanged — initializeStorageBuckets is called
// on startup and simply logs that PC storage is active.

export async function initializeStorageBuckets(): Promise<void> {
  logger.info(
    "Storage: PC local storage active (AI service /media) — Supabase disabled",
  );
}
