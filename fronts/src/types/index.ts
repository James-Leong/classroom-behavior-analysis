export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface FaceDetection {
  bbox: BBox;
  quality: number;
  similarity: number;
  identity: string;
  body_bbox?: BBox;
  track_display_identity?: string;
  track_is_locked?: boolean;
  face_detection_status?: string;
}

export interface FrameData {
  timestamp: number; // in seconds or frame index? usually seconds
  detections: FaceDetection[];
}

export interface Tracklet {
  track_id: string;
  frames_count: number;
  resolved_identity: string;
  identities_freq: Record<string, number>;
  start_time?: number;
  end_time?: number;
}

export interface FaceManifest {
  meta: {
    video_path: string;
    fps: number;
    frame_count: number;
    duration: number;
  };
  tracklets: Tracklet[];
  chunks: Array<{
    chunk_id: number;
    start_time: number;
    end_time: number;
    file: string;
  }>;
}

export interface FaceResults {
  meta: {
    video_path: string;
    fps: number;
    frame_count: number;
    duration: number;
  };
  frames: Record<string, FrameData>; // key is frame index or timestamp string
  tracklets: Tracklet[];
}

export interface BehaviorSegment {
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  label: string;
}

export interface StudentBehaviorStats {
  total_observed_seconds: number;
  behaviors: Record<string, {
    total_seconds: number;
    ratio: number;
    segments: BehaviorSegment[];
  }>;
}

export interface BehaviorStats {
  timebase: {
    fps: number;
    used_frame_interval: number;
    sample_dt_seconds: number;
  };
  denominator: number;
  by_student: Record<string, StudentBehaviorStats>;
}

export interface DebugTraceItem {
  frame: number;
  name: string;
  raw_scores: Record<string, number>;
  ema_scores: Record<string, number>;
  label: string;
  gating: {
    gated: boolean;
    top1_prob: number;
    margin: number;
    top1_label: string;
  };
  bbox: number[];
}

export type DebugTrace = DebugTraceItem[];

export interface AppState {
  currentTime: number;
  isPlaying: boolean;
  duration: number;
  playbackRate: number;
}
