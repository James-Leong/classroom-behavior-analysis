import type { FaceResults, BehaviorStats, FaceDetection, DebugTrace } from '@/types';

export const generateMockData = (): { faceResults: FaceResults, behaviorStats: BehaviorStats, debugTrace: DebugTrace } => {
  const fps = 30;
  const duration = 600; // 10 minutes
  const students = ['Alice', 'Bob', 'Charlie', 'David', 'Eve'];
  const behaviors = ['listening', 'reading', 'writing', 'raising_hand', 'distracted', 'using_device'];

  // Mock Face Results
  const frames: Record<string, any> = {};
  for (let i = 0; i < duration * fps; i += 5) { // every 5th frame to save space
    const time = i / fps;
    const detections: FaceDetection[] = students.map((id, idx) => ({
      bbox: {
        x1: 100 + idx * 150 + Math.sin(time) * 10,
        y1: 200 + Math.cos(time) * 5,
        x2: 200 + idx * 150 + Math.sin(time) * 10,
        y2: 350 + Math.cos(time) * 5,
      },
      quality: 0.9,
      similarity: 0.8,
      identity: id,
      track_display_identity: id,
      track_is_locked: true,
      body_bbox: {
        x1: 80 + idx * 150 + Math.sin(time) * 10,
        y1: 350 + Math.cos(time) * 5,
        x2: 220 + idx * 150 + Math.sin(time) * 10,
        y2: 600 + Math.cos(time) * 5,
      }
    }));
    frames[i.toString()] = { timestamp: time, detections };
  }

  const faceResults: FaceResults = {
    meta: {
      video_path: 'mock_video.mp4',
      fps,
      frame_count: duration * fps,
      duration,
    },
    frames,
    tracklets: []
  };

  // Mock Behavior Stats
  const by_student: Record<string, any> = {};
  students.forEach(id => {
    const studentBehaviors: Record<string, any> = {};
    let currentTime = 0;
    
    // Randomly assign segments
    while (currentTime < duration) {
      const label = behaviors[Math.floor(Math.random() * behaviors.length)];
      const segDuration = 5 + Math.random() * 20;
      const endTime = Math.min(currentTime + segDuration, duration);
      
      if (!studentBehaviors[label]) {
        studentBehaviors[label] = { total_seconds: 0, ratio: 0, segments: [] };
      }
      
      studentBehaviors[label].segments.push({
        start_time: currentTime,
        end_time: endTime,
        start_frame: currentTime * fps,
        end_frame: endTime * fps,
        label
      });
      studentBehaviors[label].total_seconds += (endTime - currentTime);
      
      currentTime = endTime;
    }
    
    // Calculate ratios
    Object.values(studentBehaviors).forEach((b: any) => {
      b.ratio = b.total_seconds / duration;
    });

    by_student[id] = {
      total_observed_seconds: duration,
      behaviors: studentBehaviors
    };
  });

  const behaviorStats: BehaviorStats = {
    timebase: { fps, used_frame_interval: 1, sample_dt_seconds: 1/fps },
    denominator: duration,
    by_student
  };

  // Mock Debug Trace
  const per_student_scores_raw: Record<string, any> = {};
  const per_student_scores_ema: Record<string, any> = {};
  
  students.forEach(id => {
    per_student_scores_raw[id] = {};
    per_student_scores_ema[id] = {};
    
    behaviors.forEach(label => {
      per_student_scores_raw[id][label] = {};
      per_student_scores_ema[id][label] = {};
      
      // Generate some noisy scores
      for (let i = 0; i < duration * fps; i += 30) { // every 1s
         const base = Math.random();
         per_student_scores_raw[id][label][i] = base;
         per_student_scores_ema[id][label][i] = base * 0.8 + 0.1; // smoothed
      }
    });
  });

  const debugTrace: DebugTrace = {
    per_student_scores_raw,
    per_student_scores_ema,
    per_clip_inference: [] // Populate if needed
  };

  return { faceResults, behaviorStats, debugTrace };
};
