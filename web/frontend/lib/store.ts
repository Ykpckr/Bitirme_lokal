import { create } from "zustand";

export type Theme = "light" | "dark";

interface AppState {
  theme: Theme;
  debugMode: boolean;
  normalUrl: string | null;
  debugUrl: string | null;
  mapUrl: string | null;
  videoPath: string | null;
  videoName: string | null;
  videoId: number | null;
  frameData: any[];
  currentFrame: number;
  currentTime: number;
  isPlaying: boolean;
  isProcessingFrame: boolean;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
  setDebugMode: (enabled: boolean) => void;
  selectVideo: (video: { normalUrl: string; debugUrl: string; mapUrl?: string | null; path: string; name: string; video_id?: number } | null) => void;
  addFrameData: (data: any) => void;
  clearFrameData: () => void;
  setCurrentFrame: (frame: number) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setIsProcessingFrame: (isProcessing: boolean) => void;
}
export const useAppStore = create<AppState>((set, get) => ({
  // Use deterministic defaults to avoid SSR/CSR hydration mismatches.
  // Real values are applied client-side in ThemeProvider via localStorage.
  theme: "dark",
  debugMode: false,
  normalUrl: null,
  debugUrl: null,
  mapUrl: null,
  videoPath: null,
  videoName: null,
  videoId: null,
  frameData: [],
  currentFrame: 0,
  currentTime: 0,
  isPlaying: false,
  isProcessingFrame: false,
  setTheme: (theme) => {
    set({ theme });
    if (typeof window !== "undefined") {
      localStorage.setItem("fomac-theme", theme);
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  },
  toggleTheme: () =>
    set((state) => {
      const newTheme = state.theme === "light" ? "dark" : "light";
      if (typeof window !== "undefined") {
        localStorage.setItem("fomac-theme", newTheme);
        document.documentElement.classList.toggle("dark", newTheme === "dark");
      }
      return { theme: newTheme };
    }),
  setDebugMode: (enabled) => {
    set({ debugMode: enabled });
    if (typeof window !== "undefined") {
      localStorage.setItem("fomac-debug", enabled.toString());
    }
  },
  selectVideo: (video) => {
    if (video) {
      const { normalUrl, debugUrl, mapUrl, path, name, video_id } = video as any;
      set({ 
        normalUrl, 
        debugUrl, 
        mapUrl: mapUrl ?? null,
        videoPath: path, 
        videoName: name, 
        videoId: video_id ?? null,
        frameData: [], 
        currentFrame: 0, 
        currentTime: 0 
      });
    } else {
      // Clear video related state
      set({ 
        normalUrl: null, 
        debugUrl: null, 
        mapUrl: null,
        videoPath: null, 
        videoName: null, 
        videoId: null,
        frameData: [], 
        currentFrame: 0, 
        currentTime: 0 
      });
    }
  },
  addFrameData: (data) =>
    set((state) => {
      // Prevent duplicates
      if (state.frameData.some(frame => frame.frame_id === data.frame_id)) {
        return { frameData: state.frameData.map(frame => frame.frame_id === data.frame_id ? data : frame) };
      }
      return { frameData: [...state.frameData, data].sort((a, b) => a.frame_id - b.frame_id) };
    }),
  clearFrameData: () => set({ frameData: [] }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setCurrentTime: (time) => set({ currentTime: time }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setIsProcessingFrame: (isProcessing) => set({ isProcessingFrame: isProcessing }),
}));
