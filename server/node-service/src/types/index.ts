// Camera types
export interface Camera {
  id: string;
  name: string;
  location: string;
  rtsp_url: string;
  status: 'online' | 'offline' | 'error';
  fps: number;
  resolution: string;
  pixels_per_meter: number;
  speed_limit: number;
  detection_confidence: number;
  created_at: Date;
  updated_at: Date;
}

// Detection types
export interface Detection {
  id?: number;
  camera_id: string;
  timestamp: Date;
  vehicle_type: 'car' | 'truck' | 'bus' | 'motorcycle' | 'bicycle' | 'unknown';
  speed?: number;
  confidence: number;
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
  direction?: string;
  lane_number?: number;
}

// Bounding box
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

// Incident types
export interface Incident {
  id?: number;
  camera_id: string;
  timestamp: Date;
  incident_type: 'crash' | 'speeding' | 'wrong_way' | 'stopped_vehicle' | 'congestion' | 'illegal_parking' | 'other';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description?: string;
  image_url?: string;
  video_url?: string;
  confidence?: number;
  status: 'active' | 'resolved' | 'false_alarm' | 'investigating';
  resolved_at?: Date;
  resolved_by?: string;
  notes?: string;
  metadata?: any;
}

// Analytics types
export interface HourlyAnalytics {
  id?: number;
  camera_id: string;
  hour_timestamp: Date;
  total_vehicles: number;
  avg_speed?: number;
  max_speed?: number;
  min_speed?: number;
  car_count: number;
  truck_count: number;
  bus_count: number;
  motorcycle_count: number;
  bicycle_count: number;
  incident_count: number;
  speeding_violations: number;
  peak_flow_minute?: number;
}

// Recording types
export interface Recording {
  id: string;
  camera_id: string;
  start_time: Date;
  end_time?: Date;
  duration_seconds?: number;
  video_url?: string;
  thumbnail_url?: string;
  file_size_mb?: number;
  format: string;
  resolution?: string;
  fps?: number;
  status: 'recording' | 'completed' | 'failed' | 'deleted';
  error_message?: string;
  vehicle_count: number;
  incident_count: number;
}

// AI Service types
export interface AIDetectionRequest {
  image: string; // base64 encoded image
  camera_id: string;
  timestamp: string;
  confidence_threshold?: number;
}

export interface AIDetectionResult {
  detections: Array<{
    class: string;
    confidence: number;
    bbox: BoundingBox;
    speed?: number;
  }>;
  incidents: Array<{
    type: string;
    severity: string;
    confidence: number;
    description: string;
  }>;
  processing_time_ms: number;
}

// WebSocket event types
export interface WSDetectionEvent {
  type: 'detection';
  data: Detection;
}

export interface WSIncidentEvent {
  type: 'incident';
  data: Incident;
}

export interface WSCameraStatusEvent {
  type: 'camera_status';
  data: {
    camera_id: string;
    status: 'online' | 'offline' | 'error';
    message?: string;
  };
}

export type WebSocketEvent = WSDetectionEvent | WSIncidentEvent | WSCameraStatusEvent;

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Stream types
export interface StreamConfig {
  camera_id: string;
  rtsp_url: string;
  fps: number;
  processing_rate: number; // Process every Nth frame
}
