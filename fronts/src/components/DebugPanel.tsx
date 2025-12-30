import React, { useMemo } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { BEHAVIOR_COLORS, BEHAVIOR_LABELS } from '@/constants/behaviors';

const DebugPanel: React.FC = () => {
  const { debugTrace, selectedStudentId, currentTime, faceResults } = useAppStore();

  const scores = useMemo(() => {
    if (!debugTrace || !selectedStudentId || !faceResults) return null;

    const fps = faceResults.meta.fps || 30;
    const frameIndex = Math.floor(currentTime * fps);
    
    // Find closest entry for this student
    // debugTrace is now an array of objects
    // We want entry where name === selectedStudentId and frame is closest to frameIndex
    
    // 1. Filter by student
    const studentEntries = debugTrace.filter(item => item.name === selectedStudentId);
    if (studentEntries.length === 0) return null;

    // 2. Find closest frame
    const closestEntry = studentEntries.reduce((prev, curr) => 
      Math.abs(curr.frame - frameIndex) < Math.abs(prev.frame - frameIndex) ? curr : prev
    );

    // Only show if within 1 second (30 frames)
    if (Math.abs(closestEntry.frame - frameIndex) > 30) return null;

    return closestEntry.ema_scores;
  }, [debugTrace, selectedStudentId, currentTime, faceResults]);

  if (!scores) return null;

  return (
    <div className="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase">Real-time Inference (EMA)</h3>
      <div className="space-y-3">
        {Object.entries(scores).sort(([,a], [,b]) => b - a).map(([label, score]) => (
          <div key={label}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-gray-400 capitalize">{BEHAVIOR_LABELS[label] || label.replace('_', ' ')}</span>
              <span className="text-gray-200 font-mono">{(score * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="h-2 rounded-full transition-all duration-300 ease-out"
                style={{ 
                  width: `${score * 100}%`,
                  backgroundColor: BEHAVIOR_COLORS[label] || '#4b5563'
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DebugPanel;
