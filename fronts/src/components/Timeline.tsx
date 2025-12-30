import React, { useMemo } from 'react';
import { useAppStore } from '@/store/useAppStore';
import type { BehaviorSegment } from '@/types';
import { BEHAVIOR_COLORS, BEHAVIOR_LABELS } from '@/constants/behaviors';

const Timeline: React.FC = () => {
  const { behaviorStats, duration, currentTime, setCurrentTime, selectedStudentId } = useAppStore();

  const segments = useMemo(() => {
    if (!behaviorStats) return [];
    
    // If a student is selected, show their segments
    // If not, maybe show aggregate or just nothing/all (too crowded)
    // Let's show selected student's segments flattened
    
    if (!selectedStudentId) return [];

    const studentStats = behaviorStats.by_student[selectedStudentId];
    if (!studentStats) return [];

    const allSegments: BehaviorSegment[] = [];
    Object.entries(studentStats.behaviors).forEach(([label, data]) => {
      data.segments.forEach(seg => {
        allSegments.push({ ...seg, label });
      });
    });

    return allSegments;
  }, [behaviorStats, selectedStudentId]);

  if (!selectedStudentId) {
    return (
      <div className="w-full h-24 bg-gray-900 rounded-lg flex items-center justify-center text-gray-500">
        Select a student to view behavior timeline
      </div>
    );
  }

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = x / rect.width;
    const time = percentage * (duration || 1);
    setCurrentTime(time);
  };

  return (
    <div className="w-full bg-gray-900 rounded-lg p-4">
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm font-semibold text-gray-300">Behavior Timeline ({selectedStudentId})</h3>
        <div className="flex gap-2 text-xs">
          {Object.entries(BEHAVIOR_COLORS).map(([label, color]) => (
            <div key={label} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }}></span>
              <span className="text-gray-400">{BEHAVIOR_LABELS[label] || label}</span>
            </div>
          ))}
        </div>
      </div>
      
      <div 
        className="relative w-full h-12 bg-gray-800 rounded cursor-pointer overflow-hidden"
        onClick={handleTimelineClick}
      >
        {/* Cursor Indicator */}
        <div 
          className="absolute top-0 bottom-0 w-0.5 bg-white z-10"
          style={{ left: `${(currentTime / (duration || 1)) * 100}%` }}
        />

        {/* Segments */}
        {segments.map((seg, idx) => {
          const startPct = (seg.start_time / (duration || 1)) * 100;
          const widthPct = ((seg.end_time - seg.start_time) / (duration || 1)) * 100;
          
          return (
            <div
              key={idx}
              className="absolute top-1 bottom-1 rounded-sm opacity-80 hover:opacity-100 transition-opacity"
              style={{
                left: `${startPct}%`,
                width: `${widthPct}%`,
                backgroundColor: BEHAVIOR_COLORS[seg.label] || '#999',
              }}
              title={`${BEHAVIOR_LABELS[seg.label] || seg.label}: ${seg.start_time.toFixed(1)}s - ${seg.end_time.toFixed(1)}s`}
            />
          );
        })}
      </div>
    </div>
  );
};

export default Timeline;
