import { create } from 'zustand';
import type { FaceResults, BehaviorStats, DebugTrace, FaceManifest } from '@/types';
import { generateMockData } from '@/utils/mockData';

interface AppStore {
  // Data
  faceResults: FaceResults | null;
  behaviorStats: BehaviorStats | null;
  debugTrace: DebugTrace | null;
  videoUrl: string;
  faceManifest: FaceManifest | null;
  loadedChunks: number[];
  loadingChunks: number[];

  // State
  isLoading: boolean;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  playbackRate: number;
  selectedStudentId: string | null;

  // Actions
  setFaceResults: (data: FaceResults) => void;
  setBehaviorStats: (data: BehaviorStats) => void;
  setDebugTrace: (data: DebugTrace) => void;
  setVideoUrl: (url: string) => void;
  
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (isPlaying: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  setSelectedStudentId: (id: string | null) => void;

  loadData: () => Promise<void>;
  loadChunk: (chunkId: number) => Promise<void>;
  checkAndLoadChunk: (time: number) => void;
}

export const useAppStore = create<AppStore>((set, get) => ({
  faceResults: null,
  behaviorStats: null,
  debugTrace: null,
  videoUrl: '', 
  faceManifest: null,
  loadedChunks: [],
  loadingChunks: [],

  isLoading: false,
  currentTime: 0,
  duration: 0,
  isPlaying: false,
  playbackRate: 1,
  selectedStudentId: null,

  setFaceResults: (data) => set({ faceResults: data }),
  setBehaviorStats: (data) => set({ behaviorStats: data }),
  setDebugTrace: (data) => set({ debugTrace: data }),
  setVideoUrl: (url) => set({ videoUrl: url }),

  setCurrentTime: (time) => {
    set({ currentTime: time });
    get().checkAndLoadChunk(time);
  },
  setDuration: (duration) => set({ duration }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),
  setPlaybackRate: (rate) => set({ playbackRate: rate }),
  setSelectedStudentId: (id) => set({ selectedStudentId: id }),

  loadChunk: async (chunkId: number) => {
    const { faceManifest, loadedChunks, loadingChunks, faceResults } = get();
    if (!faceManifest || loadedChunks.includes(chunkId) || loadingChunks.includes(chunkId)) {
        return;
    }

    const chunkInfo = faceManifest.chunks.find(c => c.chunk_id === chunkId);
    if (!chunkInfo) return;

    set({ loadingChunks: [...loadingChunks, chunkId] });

    try {
        const basePath = import.meta.env.BASE_URL;
        const dataPath = basePath === '/' ? '/data/' : `${basePath}data/`;
        
        console.log(`Loading chunk ${chunkId} from ${chunkInfo.file}`);
        const chunkRes = await fetch(`${dataPath}outputs/${chunkInfo.file}?t=${Date.now()}`).then(r => {
            if (!r.ok) throw new Error(`Chunk ${chunkId} not found`);
            return r.json();
        });

        // Merge frames
        const newFrames = chunkRes.frames || {};
        const updatedFaceResults = {
            ...faceResults!,
            frames: { ...faceResults!.frames, ...newFrames }
        };

        set({ 
            faceResults: updatedFaceResults,
            loadedChunks: [...get().loadedChunks, chunkId],
            loadingChunks: get().loadingChunks.filter(id => id !== chunkId)
        });
        console.log(`Loaded chunk ${chunkId}, total frames: ${Object.keys(updatedFaceResults.frames).length}`);

    } catch (e) {
        console.error(`Failed to load chunk ${chunkId}:`, e);
        set({ loadingChunks: get().loadingChunks.filter(id => id !== chunkId) });
    }
  },

  checkAndLoadChunk: (time: number) => {
      const { faceManifest, loadedChunks, loadingChunks } = get();
      if (!faceManifest) return;
      
      const chunk = faceManifest.chunks.find(c => time >= c.start_time && time < c.end_time);
      if (chunk) {
          if (!loadedChunks.includes(chunk.chunk_id) && !loadingChunks.includes(chunk.chunk_id)) {
              get().loadChunk(chunk.chunk_id);
          }
          // Preload next chunk if close to end? Optional optimization.
      }
  },

  loadData: async () => {
    set({ isLoading: true });
    try {
      const basePath = import.meta.env.BASE_URL;
      console.log('Loading data from base path:', basePath);
      
      // Try to load real data first, fallback to mock
      try {
        const dataPath = basePath === '/' ? '/data/' : `${basePath}data/`;
        console.log('Attempting to fetch data from:', dataPath);

        // Load Manifest instead of full file
        const manifestRes = await fetch(`${dataPath}outputs/face_manifest.json?t=${Date.now()}`).then(r => {
           if (!r.ok) throw new Error(`Face manifest not found at ${r.url}`);
           return r.json();
        });

        const behaviorRes = await fetch(`${dataPath}outputs/behavior_finetuned.json?t=${Date.now()}`).then(r => {
           if (!r.ok) throw new Error(`Behavior stats not found at ${r.url}`);
           return r.json();
        });
        
        let debugTraceRes = null;
        try {
          debugTraceRes = await fetch(`${dataPath}outputs/debug_trace.json`).then(r => {
            if (!r.ok) throw new Error("Not found");
            return r.json();
          });
        } catch (e) {
          console.warn("Debug trace not found, skipping", e);
        }
        
        // Initialize with empty frames but valid meta/tracklets
        const initialFaceResults: FaceResults = {
            meta: manifestRes.meta,
            tracklets: manifestRes.tracklets,
            frames: {}
        };

        set({ 
          faceManifest: manifestRes,
          faceResults: initialFaceResults, 
          behaviorStats: behaviorRes,
          debugTrace: debugTraceRes,
          videoUrl: `${dataPath}video/20251115_1h.mp4`,
          duration: manifestRes.meta.duration 
        });

        // Load initial chunk (time 0)
        get().checkAndLoadChunk(0);

      } catch (e) {
        console.warn("Using mock data because real data load failed:", e);
        const { faceResults, behaviorStats, debugTrace } = generateMockData();
        set({ 
          faceResults, 
          behaviorStats,
          debugTrace,
          videoUrl: 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', // Demo video
          duration: faceResults.meta.duration 
        });
      }
      
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      set({ isLoading: false });
    }
  },
}));
