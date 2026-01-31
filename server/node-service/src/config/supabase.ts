import { createClient, SupabaseClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';
import { logger } from './logger';

dotenv.config();

const supabaseUrl = process.env.SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing Supabase environment variables');
}

// Create Supabase client with service role key (bypasses RLS)
export const supabase: SupabaseClient = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
  },
});

// Storage bucket names
export const STORAGE_BUCKETS = {
  INCIDENTS: process.env.SUPABASE_BUCKET_INCIDENTS || 'incidents',
  RECORDINGS: process.env.SUPABASE_BUCKET_RECORDINGS || 'recordings',
};

/**
 * Upload image to Supabase Storage
 * @param bucket - Storage bucket name
 * @param path - File path in bucket
 * @param file - File buffer
 * @param contentType - MIME type
 * @returns Public URL of uploaded file
 */
export async function uploadImage(
  bucket: string,
  path: string,
  file: Buffer,
  contentType: string = 'image/jpeg'
): Promise<string | null> {
  try {
    const { data, error } = await supabase.storage.from(bucket).upload(path, file, {
      contentType,
      upsert: false,
    });

    if (error) {
      logger.error('Supabase upload error:', error);
      return null;
    }

    // Get public URL
    const {
      data: { publicUrl },
    } = supabase.storage.from(bucket).getPublicUrl(data.path);

    logger.info(`Image uploaded to Supabase: ${publicUrl}`);
    return publicUrl;
  } catch (error) {
    logger.error('Error uploading to Supabase:', error);
    return null;
  }
}

/**
 * Upload video to Supabase Storage
 * @param bucket - Storage bucket name
 * @param path - File path in bucket
 * @param file - File buffer
 * @returns Public URL of uploaded file
 */
export async function uploadVideo(
  bucket: string,
  path: string,
  file: Buffer
): Promise<string | null> {
  return uploadImage(bucket, path, file, 'video/mp4');
}

/**
 * Delete file from Supabase Storage
 * @param bucket - Storage bucket name
 * @param path - File path in bucket
 * @returns Success status
 */
export async function deleteFile(bucket: string, path: string): Promise<boolean> {
  try {
    const { error } = await supabase.storage.from(bucket).remove([path]);

    if (error) {
      logger.error('Supabase delete error:', error);
      return false;
    }

    logger.info(`File deleted from Supabase: ${path}`);
    return true;
  } catch (error) {
    logger.error('Error deleting from Supabase:', error);
    return false;
  }
}

/**
 * Initialize Supabase storage buckets
 */
export async function initializeStorageBuckets(): Promise<void> {
  try {
    // Check if buckets exist
    const { data: buckets, error } = await supabase.storage.listBuckets();

    if (error) {
      logger.error('Error listing buckets:', error);
      return;
    }

    const bucketNames = buckets.map((b) => b.name);

    // Create incidents bucket if it doesn't exist
    if (!bucketNames.includes(STORAGE_BUCKETS.INCIDENTS)) {
      const { error: createError } = await supabase.storage.createBucket(
        STORAGE_BUCKETS.INCIDENTS,
        {
          public: true,
          fileSizeLimit: 10485760, // 10MB
        }
      );
      if (createError) {
        logger.error('Error creating incidents bucket:', createError);
      } else {
        logger.info('Incidents bucket created');
      }
    }

    // Create recordings bucket if it doesn't exist
    if (!bucketNames.includes(STORAGE_BUCKETS.RECORDINGS)) {
      const { error: createError } = await supabase.storage.createBucket(
        STORAGE_BUCKETS.RECORDINGS,
        {
          public: true,
          fileSizeLimit: 524288000, // 500MB
        }
      );
      if (createError) {
        logger.error('Error creating recordings bucket:', createError);
      } else {
        logger.info('Recordings bucket created');
      }
    }

    logger.info('Supabase storage buckets initialized');
  } catch (error) {
    logger.error('Error initializing storage buckets:', error);
  }
}
