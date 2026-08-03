-- Road Sentinel MySQL Database Schema
-- Database: road_sentinel (Aiven) / self-hosted (post Phase 0.5)
--
-- GENERATED REFERENCE — this file is a static export of the schema that is
-- actually applied at every Node service startup by
-- server/node-service/src/database/migrate.ts (runMigrations()). That file
-- is the authoritative source; if the two ever disagree again, trust
-- migrate.ts and regenerate this file to match it, not the other way around.
--
-- Drop tables if exists (for clean manual setup — migrate.ts itself uses
-- CREATE TABLE IF NOT EXISTS and never drops anything)
DROP TABLE IF EXISTS recordings;
DROP TABLE IF EXISTS hourly_analytics;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS cameras;

-- ============================================
-- CAMERAS TABLE
-- ============================================
CREATE TABLE cameras (
  id                   VARCHAR(50)  NOT NULL,
  name                 VARCHAR(100) NOT NULL,
  location             VARCHAR(200) NOT NULL,
  rtsp_url             VARCHAR(500),
  status               ENUM('online','offline','error') NOT NULL DEFAULT 'offline',
  fps                  INT          NOT NULL DEFAULT 30,
  resolution           VARCHAR(20)  NOT NULL DEFAULT '1920x1080',
  pixels_per_meter     FLOAT        NOT NULL DEFAULT 8.0 COMMENT 'For speed calculation calibration (fallback when uncalibrated)',
  speed_limit          FLOAT        NOT NULL DEFAULT 60.0 COMMENT 'Speed limit in km/h',
  detection_confidence FLOAT        NOT NULL DEFAULT 0.5 COMMENT 'YOLO confidence threshold',
  homography_points    JSON         NULL COMMENT 'Perspective calibration from the Calibration Tool: {image_points:[[x,y]x4], real_points:[[x,y]x4] in meters}. NULL = uncalibrated.',
  created_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at           TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- DETECTIONS TABLE (Vehicle Detections)
-- ============================================
CREATE TABLE detections (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_id    VARCHAR(50)     NOT NULL,
  timestamp    TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  vehicle_type ENUM('car','truck','bus','motorcycle','bicycle','unknown') NOT NULL,
  speed        FLOAT COMMENT 'Speed in km/h',
  confidence   FLOAT           NOT NULL COMMENT 'Detection confidence 0-1',
  bbox_x       FLOAT           NOT NULL COMMENT 'Bounding box X coordinate',
  bbox_y       FLOAT           NOT NULL COMMENT 'Bounding box Y coordinate',
  bbox_width   FLOAT           NOT NULL COMMENT 'Bounding box width',
  bbox_height  FLOAT           NOT NULL COMMENT 'Bounding box height',
  direction    VARCHAR(50) COMMENT 'north, south, east, west',
  lane_number  INT COMMENT 'Detected lane number',
  PRIMARY KEY (id),
  INDEX idx_det_cam_time (camera_id, timestamp),
  CONSTRAINT fk_det_camera FOREIGN KEY (camera_id)
    REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- INCIDENTS TABLE
-- ============================================
CREATE TABLE incidents (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_id     VARCHAR(50)     NOT NULL,
  camera_name   VARCHAR(200),
  timestamp     TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  incident_type ENUM('crash','speeding','wrong_way','stopped_vehicle','congestion','illegal_parking','other') NOT NULL,
  severity      ENUM('low','medium','high','critical') NOT NULL,
  title         VARCHAR(200)    NOT NULL,
  description   TEXT,
  image_url     VARCHAR(1000) COMMENT 'AI service local media URL (Cloudflare Tunnel)',
  video_url     VARCHAR(1000) COMMENT 'AI service local media URL (Cloudflare Tunnel)',
  confidence    FLOAT COMMENT 'Incident detection confidence 0-1',
  status        ENUM('active','resolved','false_alarm','investigating') NOT NULL DEFAULT 'active',
  resolved_at   TIMESTAMP       NULL DEFAULT NULL,
  resolved_by   VARCHAR(100),
  notes         TEXT,
  metadata      JSON COMMENT 'Additional incident data',
  PRIMARY KEY (id),
  INDEX idx_inc_cam_time (camera_id, timestamp),
  INDEX idx_inc_status   (status),
  CONSTRAINT fk_inc_camera FOREIGN KEY (camera_id)
    REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- HOURLY_ANALYTICS TABLE (Aggregated Statistics)
-- ============================================
CREATE TABLE hourly_analytics (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  camera_id           VARCHAR(50)     NOT NULL,
  hour_timestamp      TIMESTAMP       NOT NULL COMMENT 'Start of the hour',
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
  peak_flow_minute    INT COMMENT 'Minute with highest traffic',
  PRIMARY KEY (id),
  UNIQUE KEY uq_cam_hour (camera_id, hour_timestamp),
  INDEX idx_ana_cam_hour (camera_id, hour_timestamp),
  CONSTRAINT fk_ana_camera FOREIGN KEY (camera_id)
    REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- RECORDINGS TABLE (Phase 2 — populated by camera_sender.py --record, opt-in)
-- ============================================
CREATE TABLE recordings (
  id                VARCHAR(36)     NOT NULL,
  camera_id         VARCHAR(50)     NOT NULL,
  start_time        TIMESTAMP       NOT NULL,
  end_time          TIMESTAMP       NULL,
  duration_seconds  INT,
  video_url         VARCHAR(1000) COMMENT 'AI service local media URL',
  thumbnail_url     VARCHAR(1000) COMMENT 'AI service local media URL',
  file_size_mb       FLOAT,
  format            VARCHAR(20)     NOT NULL DEFAULT 'mp4',
  resolution        VARCHAR(20),
  fps               INT,
  status            ENUM('recording','completed','failed','deleted') NOT NULL DEFAULT 'recording',
  error_message     TEXT,
  vehicle_count     INT             NOT NULL DEFAULT 0 COMMENT 'Frames with >=1 vehicle detected during the segment, not unique vehicles',
  incident_count    INT             NOT NULL DEFAULT 0,
  created_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  INDEX idx_rec_cam_time (camera_id, start_time),
  INDEX idx_rec_status   (status),
  CONSTRAINT fk_rec_camera FOREIGN KEY (camera_id)
    REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================
-- SAMPLE DATA (Optional — for manual/local testing only.
-- The real seed on every startup is server/node-service/src/database/seed.ts,
-- which seeds CAM-A-001 / CAM-B-002 with whatever CAM_A_RTSP/CAM_B_RTSP are
-- set in .env — the two hardcoded RTSP URLs below are illustrative only.)
-- ============================================

-- INSERT INTO cameras (id, name, location, rtsp_url, status, fps, resolution, pixels_per_meter, speed_limit)
-- VALUES
--   ('CAM-A-001', 'Camera A', 'Busay Blind Curve — Approach', 'rtsp://192.168.8.104:554/cam/realmonitor', 'offline', 15, '640x480', 8.0, 40),
--   ('CAM-B-002', 'Camera B', 'Busay Blind Curve — Exit',     'rtsp://192.168.8.108:554/cam/realmonitor', 'offline', 15, '640x480', 8.0, 40);

-- ============================================
-- USEFUL QUERIES
-- ============================================

-- Get live vehicle count per camera (last 5 minutes)
-- SELECT camera_id, COUNT(*) as vehicle_count, AVG(speed) as avg_speed
-- FROM detections
-- WHERE timestamp >= NOW() - INTERVAL 5 MINUTE
-- GROUP BY camera_id;

-- Get active incidents
-- SELECT i.*, c.name as camera_name, c.location
-- FROM incidents i
-- JOIN cameras c ON i.camera_id = c.id
-- WHERE i.status = 'active'
-- ORDER BY i.timestamp DESC;

-- Get hourly traffic statistics for today
-- SELECT camera_id, hour_timestamp, total_vehicles, avg_speed, incident_count
-- FROM hourly_analytics
-- WHERE DATE(hour_timestamp) = CURDATE()
-- ORDER BY hour_timestamp DESC;

-- Get speeding violations (vehicles exceeding camera speed limit)
-- SELECT d.*, c.name as camera_name, c.speed_limit
-- FROM detections d
-- JOIN cameras c ON d.camera_id = c.id
-- WHERE d.speed > c.speed_limit
-- ORDER BY d.timestamp DESC
-- LIMIT 100;
