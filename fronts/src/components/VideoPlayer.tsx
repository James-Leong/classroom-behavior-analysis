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

  function drawBBox(
    ctx: CanvasRenderingContext2D,
    bbox: BBox | number[],
    color: string,
    scaleX: number,
    scaleY: number,
    label: string | null,
  ) {
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
  }

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
      const drawingCtx: CanvasRenderingContext2D = ctx;

      // Match canvas size to displayed video size
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      }

      // Clear canvas
      drawingCtx.clearRect(0, 0, canvas.width, canvas.height);

      // Calculate scale
      const scaleX = canvas.width / (videoDimensions.width || 1);
      const scaleY = canvas.height / (videoDimensions.height || 1);

      // Find current frame data using video.currentTime directly
      const fps = faceResults.meta.fps || 30;
      const currentVideoTime = video.currentTime;
      const frameIndex = Math.round(currentVideoTime * fps);
      
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
          drawBBox(drawingCtx, det.bbox, 'rgba(0, 255, 0, 0.8)', scaleX, scaleY, identity);
        });
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [faceResults, videoDimensions, selectedStudentId]); // selectedStudentId is a dependency, so effect re-runs when it changes

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
