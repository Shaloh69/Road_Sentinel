import { pool } from "../config/database";
import { logger } from "../config/logger";

// All DDL statements run in order. Each is idempotent (IF NOT EXISTS / MODIFY COLUMN).
const MIGRATIONS: string[] = [
  // ── cameras ─────────────────────────────────────────────────────────────────
  `CREATE TABLE IF NOT EXISTS cameras (
    id                   VARCHAR(50)  NOT NULL,
    name                 VARCHAR(100) NOT NULL,
    location             VARCHAR(200) NOT NULL,
    rtsp_url             VARCHAR(500),
    status               ENUM('online','offline','error') NOT NULL DEFAULT 'offline',
    fps                  INT          NOT NULL DEFAULT 30,
    resolution           VARCHAR(20)  NOT NULL DEFAULT '1920x1080',
    pixels_per_meter     FLOAT        NOT NULL DEFAULT 8.0,
    speed_limit          FLOAT        NOT NULL DEFAULT 60.0,
    detection_confidence FLOAT        NOT NULL DEFAULT 0.5,
    homography_points    JSON         NULL COMMENT 'Phase 1: {image_points:[[x,y]x4], real_points:[[x,y]x4] in meters} from the Calibration Tool. NULL = uncalibrated, falls back to pixels_per_meter.',
    created_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,

  // Add-column for cameras created before homography_points existed (CREATE
  // TABLE IF NOT EXISTS above is a no-op on an already-existing table). MySQL
  // has no ADD COLUMN IF NOT EXISTS clause (confirmed against a real 8.0.46
  // instance — it's a parse error, not a no-op) — runMigrations() below
  // catches ER_DUP_FIELDNAME (1060) for this specific statement instead, so
  // it's still idempotent on a database that already has the column.
  `ALTER TABLE cameras ADD COLUMN homography_points JSON NULL
    COMMENT 'Phase 1: {image_points:[[x,y]x4], real_points:[[x,y]x4] in meters} from the Calibration Tool. NULL = uncalibrated, falls back to pixels_per_meter.'`,

  // ── detections ───────────────────────────────────────────────────────────────
  `CREATE TABLE IF NOT EXISTS detections (
    id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    camera_id    VARCHAR(50)     NOT NULL,
    timestamp    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    vehicle_type ENUM('car','truck','bus','motorcycle','bicycle','unknown') NOT NULL,
    speed        FLOAT,
    confidence   FLOAT           NOT NULL,
    bbox_x       FLOAT           NOT NULL,
    bbox_y       FLOAT           NOT NULL,
    bbox_width   FLOAT           NOT NULL,
    bbox_height  FLOAT           NOT NULL,
    direction    VARCHAR(50),
    lane_number  INT,
    PRIMARY KEY (id),
    INDEX idx_det_cam_time (camera_id, timestamp),
    CONSTRAINT fk_det_camera FOREIGN KEY (camera_id)
      REFERENCES cameras(id) ON DELETE CASCADE
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,

  // ── incidents ────────────────────────────────────────────────────────────────
  `CREATE TABLE IF NOT EXISTS incidents (
    id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    camera_id     VARCHAR(50)     NOT NULL,
    camera_name   VARCHAR(200),
    timestamp     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    incident_type ENUM('crash','speeding','wrong_way','stopped_vehicle','congestion','illegal_parking','other') NOT NULL,
    severity      ENUM('low','medium','high','critical') NOT NULL,
    title         VARCHAR(200)    NOT NULL,
    description   TEXT,
    image_url     VARCHAR(1000),
    video_url     VARCHAR(1000),
    confidence    FLOAT,
    status        ENUM('active','resolved','false_alarm','investigating') NOT NULL DEFAULT 'active',
    resolved_at   TIMESTAMP       NULL DEFAULT NULL,
    resolved_by   VARCHAR(100),
    notes         TEXT,
    metadata      JSON,
    PRIMARY KEY (id),
    INDEX idx_inc_cam_time (camera_id, timestamp),
    INDEX idx_inc_status   (status),
    CONSTRAINT fk_inc_camera FOREIGN KEY (camera_id)
      REFERENCES cameras(id) ON DELETE CASCADE
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,

  // ── hourly_analytics ─────────────────────────────────────────────────────────
  `CREATE TABLE IF NOT EXISTS hourly_analytics (
    id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    camera_id           VARCHAR(50)     NOT NULL,
    hour_timestamp      TIMESTAMP       NOT NULL,
    total_vehicles      INT             NOT NULL DEFAULT 0,
    avg_speed           FLOAT,
    max_speed           FLOAT,
    min_speed           FLOAT,
    car_count           INT             NOT NULL DEFAULT 0,
    truck_count         INT             NOT NULL DEFAULT 0,
    bus_count           INT             NOT NULL DEFAULT 0,
    motorcycle_count    INT             NOT NULL DEFAULT 0,
    bicycle_count       INT             NOT NULL DEFAULT 0,
    incident_count      INT             NOT NULL DEFAULT 0,
    speeding_violations INT             NOT NULL DEFAULT 0,
    peak_flow_minute    INT,
    PRIMARY KEY (id),
    UNIQUE KEY uq_cam_hour (camera_id, hour_timestamp),
    INDEX idx_ana_cam_hour (camera_id, hour_timestamp),
    CONSTRAINT fk_ana_camera FOREIGN KEY (camera_id)
      REFERENCES cameras(id) ON DELETE CASCADE
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,

  // ── recordings (Phase 2) ────────────────────────────────────────────────────
  // Populated by raspi_scripts/camera/camera_sender.py --record (opt-in,
  // untested against real camera hardware as of Phase 2).
  `CREATE TABLE IF NOT EXISTS recordings (
    id                VARCHAR(36)     NOT NULL,
    camera_id         VARCHAR(50)     NOT NULL,
    start_time        TIMESTAMP       NOT NULL,
    end_time          TIMESTAMP       NULL,
    duration_seconds  INT,
    video_url         VARCHAR(1000),
    thumbnail_url     VARCHAR(1000),
    file_size_mb       FLOAT,
    format            VARCHAR(20)     NOT NULL DEFAULT 'mp4',
    resolution        VARCHAR(20),
    fps               INT,
    status            ENUM('recording','completed','failed','deleted') NOT NULL DEFAULT 'recording',
    error_message     TEXT,
    vehicle_count     INT             NOT NULL DEFAULT 0,
    incident_count    INT             NOT NULL DEFAULT 0,
    created_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_rec_cam_time (camera_id, start_time),
    INDEX idx_rec_status   (status),
    CONSTRAINT fk_rec_camera FOREIGN KEY (camera_id)
      REFERENCES cameras(id) ON DELETE CASCADE
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`,
];

// MySQL error codes that mean "this DDL statement's effect already exists" —
// safe to ignore since the equivalent CREATE TABLE IF NOT EXISTS above (or a
// prior run of this same migration) already got there.
const IDEMPOTENT_ERROR_CODES = new Set([
  "ER_DUP_FIELDNAME", // ADD COLUMN — column already exists
  "ER_DUP_KEYNAME", // ADD INDEX/KEY — index already exists
]);

export async function runMigrations(): Promise<void> {
  const conn = await pool.getConnection();
  try {
    for (const sql of MIGRATIONS) {
      try {
        await conn.execute(sql);
      } catch (err) {
        const code = (err as { code?: string }).code;
        if (code && IDEMPOTENT_ERROR_CODES.has(code)) {
          continue;
        }
        throw err;
      }
    }
    logger.info("Database migrations applied successfully");
  } catch (err) {
    logger.error("Migration failed:", err);
    throw err;
  } finally {
    conn.release();
  }
}
