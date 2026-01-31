import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';
import { logger } from '../config/logger';
import { AIDetectionRequest, AIDetectionResult } from '../types';

class AIService {
  private client: AxiosInstance;
  private baseURL: string;
  private timeout: number;

  constructor() {
    this.baseURL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
    this.timeout = parseInt(process.env.AI_SERVICE_TIMEOUT || '30000');

    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Check if AI service is healthy
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.status === 200;
    } catch (error) {
      logger.error('AI service health check failed:', error);
      return false;
    }
  }

  /**
   * Send frame to AI service for detection
   * @param imageBuffer - Image buffer (JPEG)
   * @param cameraId - Camera ID
   * @param confidenceThreshold - Detection confidence threshold
   * @returns Detection results
   */
  async detectObjects(
    imageBuffer: Buffer,
    cameraId: string,
    confidenceThreshold: number = 0.75
  ): Promise<AIDetectionResult | null> {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: `${cameraId}_${Date.now()}.jpg`,
        contentType: 'image/jpeg',
      });
      formData.append('camera_id', cameraId);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const response = await this.client.post('/api/detect', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });

      if (response.status === 200) {
        logger.debug(`AI detection successful for camera ${cameraId}`);
        return response.data;
      }

      logger.warn(`AI detection returned status ${response.status}`);
      return null;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        logger.error(
          `AI service error: ${error.message}`,
          error.response?.data
        );
      } else {
        logger.error('AI service request failed:', error);
      }
      return null;
    }
  }

  /**
   * Send frame to AI service for traffic detection only
   */
  async detectTraffic(
    imageBuffer: Buffer,
    cameraId: string,
    confidenceThreshold: number = 0.75
  ): Promise<AIDetectionResult | null> {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: `${cameraId}_${Date.now()}.jpg`,
        contentType: 'image/jpeg',
      });
      formData.append('camera_id', cameraId);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const response = await this.client.post('/api/detect/traffic', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });

      return response.status === 200 ? response.data : null;
    } catch (error) {
      logger.error('Traffic detection failed:', error);
      return null;
    }
  }

  /**
   * Send frame to AI service for incident detection only
   */
  async detectIncidents(
    imageBuffer: Buffer,
    cameraId: string,
    confidenceThreshold: number = 0.75
  ): Promise<AIDetectionResult | null> {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, {
        filename: `${cameraId}_${Date.now()}.jpg`,
        contentType: 'image/jpeg',
      });
      formData.append('camera_id', cameraId);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const response = await this.client.post('/api/detect/incidents', formData, {
        headers: {
          ...formData.getHeaders(),
        },
      });

      return response.status === 200 ? response.data : null;
    } catch (error) {
      logger.error('Incident detection failed:', error);
      return null;
    }
  }

  /**
   * Get AI service statistics
   */
  async getStats(): Promise<any> {
    try {
      const response = await this.client.get('/api/stats');
      return response.data;
    } catch (error) {
      logger.error('Failed to get AI service stats:', error);
      return null;
    }
  }
}

// Export singleton instance
export const aiService = new AIService();
