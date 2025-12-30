import React from 'react';
import { Activity } from 'lucide-react';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="h-16 bg-gray-800 border-b border-gray-700 flex items-center px-6 shadow-sm z-10">
        <div className="flex items-center gap-2">
          <Activity className="text-blue-500" />
          <h1 className="text-xl font-bold">Classroom Behavior Analysis</h1>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <span className="text-sm text-gray-400">Offline Mode</span>
        </div>
      </header>
      <main className="flex-1 flex overflow-hidden">
        {children}
      </main>
    </div>
  );
};

export default Layout;
