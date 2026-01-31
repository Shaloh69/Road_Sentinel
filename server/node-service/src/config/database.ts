import mysql from 'mysql2/promise';
import dotenv from 'dotenv';
import { logger } from './logger';

dotenv.config();

// MySQL connection pool configuration
const poolConfig: mysql.PoolOptions = {
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '3306'),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
  maxIdle: 10,
  idleTimeout: 60000,
  queueLimit: 0,
  enableKeepAlive: true,
  keepAliveInitialDelay: 0,
};

// Add SSL configuration for Aiven if enabled
if (process.env.DB_SSL === 'true') {
  poolConfig.ssl = {
    rejectUnauthorized: true,
  };
}

// Create connection pool
export const pool = mysql.createPool(poolConfig);

// Test database connection
export async function testConnection(): Promise<boolean> {
  try {
    const connection = await pool.getConnection();
    logger.info('MySQL database connected successfully');
    connection.release();
    return true;
  } catch (error) {
    logger.error('Failed to connect to MySQL database:', error);
    return false;
  }
}

// Execute query helper
export async function query<T>(sql: string, params?: any[]): Promise<T> {
  try {
    const [rows] = await pool.execute(sql, params);
    return rows as T;
  } catch (error) {
    logger.error('Database query error:', error);
    throw error;
  }
}

// Transaction helper
export async function transaction<T>(
  callback: (connection: mysql.PoolConnection) => Promise<T>
): Promise<T> {
  const connection = await pool.getConnection();
  try {
    await connection.beginTransaction();
    const result = await callback(connection);
    await connection.commit();
    return result;
  } catch (error) {
    await connection.rollback();
    throw error;
  } finally {
    connection.release();
  }
}

// Graceful shutdown
export async function closePool(): Promise<void> {
  try {
    await pool.end();
    logger.info('MySQL connection pool closed');
  } catch (error) {
    logger.error('Error closing MySQL connection pool:', error);
  }
}
