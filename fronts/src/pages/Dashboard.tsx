import React, { useEffect } from 'react';
import { useAppStore } from '@/store/useAppStore';
import VideoPlayer from '@/components/VideoPlayer';
import Timeline from '@/components/Timeline';
import DebugPanel from '@/components/DebugPanel';
import { User, BarChart2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

import { BEHAVIOR_COLORS, BEHAVIOR_LABELS } from '@/constants/behaviors';

const Dashboard: React.FC = () => {
  const { 
    behaviorStats, 
    loadData, 
    selectedStudentId, 
    setSelectedStudentId 
  } = useAppStore();

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Derive students list
  const students = behaviorStats 
    ? Object.keys(behaviorStats.by_student).sort() 
    : [];

  // Prepare overview data
  const overviewData = behaviorStats 
    ? Object.entries(behaviorStats.by_student).map(([id, stats]) => {
        const data: any = { name: id };
        Object.entries(stats.behaviors).forEach(([label, bData]) => {
          // Use mapped keys if needed, but here we just map values
          // Actually, we should check if label is in our known list
          if (label in BEHAVIOR_LABELS) {
            data[label] = bData.total_seconds;
          }
        });
        return data;
      })
    : [];

  return (
    <div className="flex w-full h-full">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Students</h2>
        </div>
        
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          <button
            onClick={() => setSelectedStudentId(null)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
              selectedStudentId === null 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-300 hover:bg-gray-700'
            }`}
          >
            <BarChart2 size={18} />
            <span className="text-sm font-medium">Class Overview</span>
          </button>

          {students.map(id => (
            <button
              key={id}
              onClick={() => setSelectedStudentId(id)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-md transition-colors ${
                selectedStudentId === id 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <User size={18} />
              <span className="text-sm font-medium truncate">{id}</span>
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto bg-gray-900 p-6">
        {!selectedStudentId ? (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold mb-4">Class Overview</h2>
            
            <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold mb-4">Behavior Distribution by Student (Seconds)</h3>
              <div style={{ width: '100%', height: 400, minHeight: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={overviewData} barGap={2} barCategoryGap="20%">
                    <XAxis 
                      dataKey="name" 
                      stroke="#9ca3af" 
                      interval={0}
                      angle={-45}
                      textAnchor="end"
                      height={80}
                    />
                    <YAxis stroke="#9ca3af" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                      formatter={(value: number, name: string) => [
                        `${Math.round(value)}s`, 
                        BEHAVIOR_LABELS[name] || name
                      ]}
                    />
                    <Legend formatter={(value) => BEHAVIOR_LABELS[value] || value} />
                    {Object.keys(BEHAVIOR_COLORS).map(key => (
                      <Bar 
                        key={key} 
                        dataKey={key} 
                        stackId="a" 
                        fill={BEHAVIOR_COLORS[key]} 
                        name={key}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">Student: {selectedStudentId}</h2>
              <div className="bg-gray-800 px-3 py-1 rounded text-sm text-gray-300">
                Total Observed: {behaviorStats?.by_student[selectedStudentId]?.total_observed_seconds.toFixed(0)}s
              </div>
            </div>

            {/* Video Player Section */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="lg:col-span-3 space-y-4">
                 <div className="w-full">
                  <VideoPlayer />
                </div>
                <div className="w-full">
                  <Timeline />
                </div>
              </div>
              
              <div className="lg:col-span-1 space-y-4">
                <DebugPanel />
              </div>
            </div>

            {/* Stats Section */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {behaviorStats?.by_student[selectedStudentId] && Object.entries(behaviorStats.by_student[selectedStudentId].behaviors).map(([label, stats]) => {
                if (!(label in BEHAVIOR_LABELS)) return null;
                return (
                  <div key={label} className="bg-gray-800 p-4 rounded-lg">
                    <h4 className="text-gray-400 text-sm capitalize mb-1">
                      {BEHAVIOR_LABELS[label] || label.replace('_', ' ')}
                    </h4>
                    <div className="text-2xl font-bold" style={{ color: BEHAVIOR_COLORS[label] }}>
                      {Math.round(stats.total_seconds)}s
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      {(stats.ratio * 100).toFixed(1)}% of time
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;

