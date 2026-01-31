import { uploadImage, uploadVideo, deleteFile, STORAGE_BUCKETS } from '../config/supabase';
import { logger } from '../config/logger';
import { v4 as uuidv4 } from 'uuid';

class StorageService {
  /**
   * Save incident snapshot to Supabase Storage
   * @param imageBuffer - Image buffer
   * @param cameraId - Camera ID
   * @param incidentType - Type of incident
   * @returns Public URL of uploaded image
   */
  async saveIncidentSnapshot(
    imageBuffer: Buffer,
    cameraId: string,
    incidentType: string
  ): Promise<string | null> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `${cameraId}/${incidentType}/${timestamp}_${uuidv4()}.jpg`;

      const url = await uploadImage(
        STORAGE_BUCKETS.INCIDENTS,
        filename,
        imageBuffer,
        'image/jpeg'
      );

      if (url) {
        logger.info(`Incident snapshot saved: ${url}`);
      }

      return url;
    } catch (error) {
      logger.error('Error saving incident snapshot:', error);
      return null;
    }
  }

  /**
   * Save incident video clip to Supabase Storage
   * @param videoBuffer - Video buffer
   * @param cameraId - Camera ID
   * @param incidentType - Type of incident
   * @returns Public URL of uploaded video
   */
  async saveIncidentVideo(
    videoBuffer: Buffer,
    cameraId: string,
    incidentType: string
  ): Promise<string | null> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `${cameraId}/${incidentType}/${timestamp}_${uuidv4()}.mp4`;

      const url = await uploadVideo(STORAGE_BUCKETS.INCIDENTS, filename, videoBuffer);

      if (url) {
        logger.info(`Incident video saved: ${url}`);
      }

      return url;
    } catch (error) {
      logger.error('Error saving incident video:', error);
      return null;
    }
  }

  /**
   * Save recording to Supabase Storage
   * @param videoBuffer - Video buffer
   * @param cameraId - Camera ID
   * @param recordingId - Recording ID
   * @returns Public URL of uploaded video
   */
  async saveRecording(
    videoBuffer: Buffer,
    cameraId: string,
    recordingId: string
  ): Promise<string | null> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `${cameraId}/${timestamp}_${recordingId}.mp4`;

      const url = await uploadVideo(STORAGE_BUCKETS.RECORDINGS, filename, videoBuffer);

      if (url) {
        logger.info(`Recording saved: ${url}`);
      }

      return url;
    } catch (error) {
      logger.error('Error saving recording:', error);
      return null;
    }
  }

  /**
   * Save recording thumbnail
   * @param imageBuffer - Thumbnail image buffer
   * @param cameraId - Camera ID
   * @param recordingId - Recording ID
   * @returns Public URL of uploaded thumbnail
   */
  async saveRecordingThumbnail(
    imageBuffer: Buffer,
    cameraId: string,
    recordingId: string
  ): Promise<string | null> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const filename = `${cameraId}/thumbnails/${timestamp}_${recordingId}.jpg`;

      const url = await uploadImage(
        STORAGE_BUCKETS.RECORDINGS,
        filename,
        imageBuffer,
        'image/jpeg'
      );

      if (url) {
        logger.info(`Recording thumbnail saved: ${url}`);
      }

      return url;
    } catch (error) {
      logger.error('Error saving recording thumbnail:', error);
      return null;
    }
  }

  /**
   * Delete incident files
   * @param imageUrl - Image URL to delete
   * @param videoUrl - Video URL to delete (optional)
   */
  async deleteIncidentFiles(imageUrl?: string, videoUrl?: string): Promise<void> {
    try {
      if (imageUrl) {
        const imagePath = this.extractPathFromUrl(imageUrl);
        if (imagePath) {
          await deleteFile(STORAGE_BUCKETS.INCIDENTS, imagePath);
        }
      }

      if (videoUrl) {
        const videoPath = this.extractPathFromUrl(videoUrl);
        if (videoPath) {
          await deleteFile(STORAGE_BUCKETS.INCIDENTS, videoPath);
        }
      }
    } catch (error) {
      logger.error('Error deleting incident files:', error);
    }
  }

  /**
   * Delete recording files
   * @param videoUrl - Recording video URL
   * @param thumbnailUrl - Thumbnail URL (optional)
   */
  async deleteRecordingFiles(videoUrl: string, thumbnailUrl?: string): Promise<void> {
    try {
      const videoPath = this.extractPathFromUrl(videoUrl);
      if (videoPath) {
        await deleteFile(STORAGE_BUCKETS.RECORDINGS, videoPath);
      }

      if (thumbnailUrl) {
        const thumbPath = this.extractPathFromUrl(thumbnailUrl);
        if (thumbPath) {
          await deleteFile(STORAGE_BUCKETS.RECORDINGS, thumbPath);
        }
      }
    } catch (error) {
      logger.error('Error deleting recording files:', error);
    }
  }

  /**
   * Extract file path from Supabase public URL
   * @param url - Supabase public URL
   * @returns File path in bucket
   */
  private extractPathFromUrl(url: string): string | null {
    try {
      // URL format: https://project.supabase.co/storage/v1/object/public/bucket-name/path/to/file
      const urlParts = url.split('/');
      const publicIndex = urlParts.indexOf('public');
      if (publicIndex !== -1 && publicIndex + 2 < urlParts.length) {
        // Skip bucket name and get the rest as path
        return urlParts.slice(publicIndex + 2).join('/');
      }
      return null;
    } catch (error) {
      logger.error('Error extracting path from URL:', error);
      return null;
    }
  }
}

// Export singleton instance
export const storageService = new StorageService();
