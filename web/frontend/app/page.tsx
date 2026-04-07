"use client";

import { VideoSelector } from "@/components/video-selector";
import { VideoPlayer } from "@/components/video-player";
import { DebugPlayer } from "@/components/debug-player";
import { useAppStore } from "@/lib/store";

export default function Home() {
  const normalUrl = useAppStore((state) => state.normalUrl);
  const debugMode = useAppStore((state) => state.debugMode);
  const debugUrl = useAppStore((state) => state.debugUrl);

  return (
    <div className="container mx-auto px-4 py-8">
      {!normalUrl ? (
        <VideoSelector />
      ) : debugMode && debugUrl ? (
        <DebugPlayer />
      ) : (
        <VideoPlayer />
      )}
    </div>
  );
}
