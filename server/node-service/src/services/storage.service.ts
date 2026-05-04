import axios from "axios";
import FormData from "form-data";
import { v4 as uuidv4 } from "uuid";
import { logger } from "../config/logger";

// All media is stored on the PC running the AI service (Cloudflare tunnel).
// The AI service exposes /api/storage/upload and /media/... for public access.
const AI_URL = (process.env.AI_SERVICE_URL || "http://localhost:8000").replace(
  /\/$/,
  "",
);

class StorageService {
  private async upload(
    buffer: Buffer,
    path: string,
    contentType: string,
  ): Promise<string | null> {
    try {
      const form = new FormData();
      form.append("file", buffer, {
        filename: path.split("/").pop()!,
        contentType,
      });
      form.append("path", path);

      const res = await axios.post(`${AI_URL}/api/storage/upload`, form, {
        headers: form.getHeaders(),
        timeout: 30_000,
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
      });

      return (res.data.url as string) ?? null;
    } catch (err) {
      logger.error(
        `PC storage upload failed (${path}): ${err instanceof Error ? err.message : String(err)}`,
      );
      return null;
    }
  }

  private async deleteByUrl(url: string): Promise<void> {
    try {
      const match = url.match(/\/media\/(.+)$/);
      if (!match) return;
      await axios.delete(`${AI_URL}/api/storage/delete`, {
        params: { path: match[1] },
        timeout: 10_000,
      });
    } catch (err) {
      logger.error(
        `PC storage delete failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  async saveIncidentSnapshot(
    imageBuffer: Buffer,
    cameraId: string,
    incidentType: string,
  ): Promise<string | null> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `incidents/${cameraId}/${incidentType}/${ts}_${uuidv4()}.jpg`;
    const url = await this.upload(imageBuffer, path, "image/jpeg");
    if (url) logger.info(`📸 Incident snapshot saved to PC storage: ${url}`);
    return url;
  }

  async saveIncidentVideo(
    videoBuffer: Buffer,
    cameraId: string,
    incidentType: string,
  ): Promise<string | null> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `incidents/${cameraId}/${incidentType}/${ts}_${uuidv4()}.mp4`;
    const url = await this.upload(videoBuffer, path, "video/mp4");
    if (url) logger.info(`🎬 Incident video saved to PC storage: ${url}`);
    return url;
  }

  async saveRecording(
    videoBuffer: Buffer,
    cameraId: string,
    recordingId: string,
  ): Promise<string | null> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `recordings/${cameraId}/${ts}_${recordingId}.mp4`;
    const url = await this.upload(videoBuffer, path, "video/mp4");
    if (url) logger.info(`🎬 Recording saved to PC storage: ${url}`);
    return url;
  }

  async saveRecordingThumbnail(
    imageBuffer: Buffer,
    cameraId: string,
    recordingId: string,
  ): Promise<string | null> {
    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const path = `recordings/${cameraId}/thumbnails/${ts}_${recordingId}.jpg`;
    const url = await this.upload(imageBuffer, path, "image/jpeg");
    if (url) logger.info(`📸 Thumbnail saved to PC storage: ${url}`);
    return url;
  }

  async deleteIncidentFiles(
    imageUrl?: string,
    videoUrl?: string,
  ): Promise<void> {
    await Promise.all([
      imageUrl ? this.deleteByUrl(imageUrl) : Promise.resolve(),
      videoUrl ? this.deleteByUrl(videoUrl) : Promise.resolve(),
    ]);
  }

  async deleteRecordingFiles(
    videoUrl: string,
    thumbnailUrl?: string,
  ): Promise<void> {
    await Promise.all([
      this.deleteByUrl(videoUrl),
      thumbnailUrl ? this.deleteByUrl(thumbnailUrl) : Promise.resolve(),
    ]);
  }
}

export const storageService = new StorageService();
