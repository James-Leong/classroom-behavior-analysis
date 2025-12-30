import React, { useRef, useEffect, useState } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { Play, Pause } from 'lucide-react';
import type { BBox } from '@/types';

const VideoPlayer: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { 
    videoUrl, 
    faceResults, 
    isPlaying, 
    currentTime, 
    duration,
    selectedStudentId,
    setIsPlaying, 
    setCurrentTime, 
    setDuration 
  } = useAppStore();

  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  // Sync store playing state with video element
  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.play().catch(e => console.error("Play error:", e));
      } else {
        videoRef.current.pause();
      }
    }
  }, [isPlaying]);

  // Sync store time with video element (if discrepancy is large)
  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.5) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
      setVideoDimensions({
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight
      });
    }
  };

  // Draw overlay using requestAnimationFrame for smoothness
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      
      // We must access the LATEST state in the render loop.
      // Since requestAnimationFrame is outside the React render cycle, 
      // relying on closure-captured 'selectedStudentId' from the effect's initial run is risky 
      // if dependencies aren't perfectly triggering re-runs.
      // However, putting selectedStudentId in the dependency array (which we did) should recreate the effect and the render closure.
      // Let's verify if the store state is actually updating correctly in the closure.
      
      if (!canvas || !video || !faceResults) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      // Match canvas size to displayed video size
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      }

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Calculate scale
      const scaleX = canvas.width / (videoDimensions.width || 1);
      const scaleY = canvas.height / (videoDimensions.height || 1);

      // Find current frame data using video.currentTime directly
      const fps = faceResults.meta.fps || 30;
      const currentVideoTime = video.currentTime;
      let frameIndex = Math.round(currentVideoTime * fps);
      
      // Direct lookup for performance, with backtracking for gaps
      // Try to find the nearest previous frame if current one is missing (up to 15 frames / 0.5s)
      let frameData = null;
      for (let offset = 0; offset < 15; offset++) {
        const checkIndex = frameIndex - offset;
        if (checkIndex < 0) break;
        
        const data = faceResults.frames[checkIndex] || faceResults.frames[checkIndex.toString()];
        if (data) {
          frameData = data;
          break;
        }
      }

      if (frameData && frameData.detections) {
        frameData.detections.forEach(det => {
          const identity = det.track_display_identity || det.identity;
          
          // Debug log (optional, remove in prod)
          // if (Math.random() < 0.01) console.log('Rendering identity:', identity, 'Selected:', selectedStudentId);

          // Filter by selected student
          // Note: identity and selectedStudentId might have different types (string vs number or undefined)
          // Safe comparison:
          const isSelected = selectedStudentId && String(identity) === String(selectedStudentId);
          
          if (selectedStudentId && !isSelected) {
            return;
          }

          // Draw Face BBox
          drawBBox(ctx, det.bbox, 'rgba(0, 255, 0, 0.8)', scaleX, scaleY, identity);
          
          // Draw Body BBox
          if (false && det.body_bbox) {
            // Check if body bbox is not too far from face bbox
            // Simple heuristic: body center should be close to face center horizontally
            // and body should be below or containing face
            
            // However, the issue might be simpler: one body bbox might be assigned to multiple faces in the raw data 
            // if the face-body matching was ambiguous.
            // Or simply visualizing all body bboxes blindly.
            
            // Since `det` is a single detection object containing both `bbox` (face) and `body_bbox` (body),
            // they SHOULD belong to the same person as per the backend logic.
            // If they look mismatched visually, it might be an issue with coordinate scaling or backend matching.
            
            // Let's assume the pairing in JSON is correct (as provided by backend).
            // We just need to draw it.
            // If the user says "not the same person", maybe the backend matching is indeed wrong for some cases,
            // OR we are drawing it wrong.
            
            // Let's check coordinates. 
            // Face: [1104, 283, 1128, 316]
            // Body: [1075, 272, 1177, 502]
            // Face is inside/near top of Body. This looks correct for "蒋军".
            
            // Another case:
            // Face: [1787, 474, 1832, 531] ("张浩")
            // Body: [1669, 535, 1873, 743]
            // Face y2=531, Body y1=535. Face is just above body? Or slightly overlapping?
            // This seems plausible.
            
            // But wait, look at the third detection in chunk 0:
            // Face: [1805, 403, 1847, 455] ("未知")
            // Body: [1669, 535, 1873, 743]
            // This is the SAME body bbox as "张浩"!
            // So two faces are claiming the same body.
            
            // If we filter by student, e.g. "张浩", we draw his face and this shared body.
            // If we filter by "未知", we draw his face and this SAME shared body.
            // Visually, if "张浩" is selected, we see a body that might be far away from his face if the matching is wrong,
            // or if it's a shared false positive.
            
            // In this specific case:
            // "张浩" Face center: [1809, 502]
            // "未知" Face center: [1826, 429]
            // Body center: [1771, 639]
            
            // "张浩" is closer to the body than "未知".
            
            // If the user sees "body bbox not corresponding to face", it implies the visual link is broken.
            // We should perhaps only draw the body bbox if we are sure it's a good match?
            // Or maybe just draw a line connecting them to show the relationship?
            // For now, let's just draw the box.
            
            // One possibility: The body bbox coordinates are in a different scale?
            // No, they look consistent (1920x1080 range).
            
            // Is it possible we are drawing the WRONG body bbox?
            // We are drawing `det.body_bbox`. It is explicitly linked in the JSON.
            
            drawBBox(ctx, det.body_bbox, 'rgba(255, 165, 0, 0.6)', scaleX, scaleY, null);
            
            // Optional: Draw a line connecting face center to body center to visualize the link
            // const faceCx = (det.bbox[0] + det.bbox[2])/2 * scaleX;
            // const faceCy = (det.bbox[1] + det.bbox[3])/2 * scaleY;
            // const bodyCx = (det.body_bbox[0] + det.body_bbox[2])/2 * scaleX;
            // const bodyCy = (det.body_bbox[1] + det.body_bbox[3])/2 * scaleY;
            // ctx.beginPath();
            // ctx.moveTo(faceCx, faceCy);
            // ctx.lineTo(bodyCx, bodyCy);
            // ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            // ctx.stroke();
          }
        });
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [faceResults, videoDimensions, selectedStudentId]); // selectedStudentId is a dependency, so effect re-runs when it changes

  const drawBBox = (
    ctx: CanvasRenderingContext2D, 
    bbox: BBox | number[], 
    color: string, 
    scaleX: number, 
    scaleY: number,
    label: string | null
  ) => {
    let x, y, w, h;

    if (Array.isArray(bbox)) {
      x = bbox[0] * scaleX;
      y = bbox[1] * scaleY;
      w = (bbox[2] - bbox[0]) * scaleX;
      h = (bbox[3] - bbox[1]) * scaleY;
    } else {
      x = bbox.x1 * scaleX;
      y = bbox.y1 * scaleY;
      w = (bbox.x2 - bbox.x1) * scaleX;
      h = (bbox.y2 - bbox.y1) * scaleY;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    if (label) {
      ctx.fillStyle = color;
      ctx.fillRect(x, y - 20, ctx.measureText(label).width + 10, 20);
      ctx.fillStyle = 'black';
      ctx.font = '12px Arial';
      ctx.fillText(label, x + 5, y - 5);
    }
  };

  const togglePlay = () => setIsPlaying(!isPlaying);

  return (
    <div className="relative flex flex-col items-center w-full max-w-4xl mx-auto bg-black rounded-lg overflow-hidden shadow-xl">
      <div ref={containerRef} className="relative w-full aspect-video">
        <video
          ref={videoRef}
          src={videoUrl}
          className="absolute top-0 left-0 w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
        />
      </div>

      {/* Controls */}
      <div className="w-full bg-gray-900 p-4 flex items-center space-x-4">
        <button onClick={togglePlay} className="p-2 hover:bg-gray-800 rounded-full text-white">
          {isPlaying ? <Pause size={24} /> : <Play size={24} />}
        </button>
        
        <div className="flex-1">
          <input
            type="range"
            min={0}
            max={duration || 100}
            value={currentTime}
            onChange={(e) => {
              const t = parseFloat(e.target.value);
              setCurrentTime(t);
              if (videoRef.current) videoRef.current.currentTime = t;
            }}
            className="w-full accent-blue-500 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        
        <div className="text-white font-mono text-sm">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>
    </div>
  );
};

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
};

export default VideoPlayer;
