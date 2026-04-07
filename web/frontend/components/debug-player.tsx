"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Play, Pause, RotateCcw, X, Loader2, Server, Framer } from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const BUFFER_SECONDS = 4; // Process first 4 seconds of video before starting playback
const PLAYBACK_RATE_SYNC_THRESHOLD = 0.5; // Start slowing down if video is 0.5s ahead
const PLAYBACK_RATE_FAST_SYNC_THRESHOLD = 1.0; // Slow down more if video is 1.0s ahead
const FPS_CALCULATION_WINDOW = 20; // Use last 20 frames to calculate FPS

type BackendStatus = "online" | "offline" | "loading";
type PlayerStatus = "buffering" | "ready" | "playing" | "paused";

interface Detection {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    class_name: string;
}

interface FrameData {
  frame_id: number;
  inference_time: number;
  detections: Detection[];
  received_time: number;
}

const DETECTION_COLORS: Record<string, { stroke: string; fill: string; label: string }> = {
  player: { stroke: "#00ff00", fill: "rgba(0, 255, 0, 0.7)", label: "Player" },
  ball: { stroke: "#ffff00", fill: "rgba(255, 255, 0, 0.7)", label: "Ball" },
  referee: { stroke: "#ff00ff", fill: "rgba(255, 0, 255, 0.7)", label: "Referee" },
};

export function DebugPlayer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mapVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement>(null);
  const lastFrameSendTime = useRef(0);
  
  const {
    debugUrl, mapUrl, selectVideo, isPlaying, setIsPlaying,
    currentTime, setCurrentTime, addFrameData, 
    frameData, setCurrentFrame, currentFrame, 
    isProcessingFrame, setIsProcessingFrame
  } = useAppStore();
  
  const [duration, setDuration] = useState(0);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("loading");
  const [playerStatus, setPlayerStatus] = useState<PlayerStatus>("buffering");
  const [currentFrameData, setCurrentFrameData] = useState<FrameData | null>(null);
  const [backendFps, setBackendFps] = useState(0);

  // --- Core Effects ---

  // Frame processing logic removed. Debug video is pre-processed and played directly.

  // For pre-processed debug videos we don't send frames to backend.
  // Instead, we simply synchronize UI state with the video element.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const syncSlave = () => {
      const mv = mapVideoRef.current;
      if (!mv) return;
      try {
        if ((mv.readyState || 0) < 1) return;
        const dt = Math.abs((mv.currentTime || 0) - (video.currentTime || 0));
        if (dt > 0.15) mv.currentTime = video.currentTime || 0;
      } catch (e) {
        // ignore
      }
    };

    const onLoaded = () => {
      setDuration(video.duration || 0);
      // sync playback position from store
      try {
        if ((videoRef.current?.readyState || 0) >= 1) {
          video.currentTime = currentTime || 0;
        }
      } catch (e) {
        // ignore
      }

      // Sync map video (if any)
      try {
        const mv = mapVideoRef.current;
        if (mv && (mv.readyState || 0) >= 1) {
          mv.currentTime = currentTime || 0;
        }
      } catch (e) {
        // ignore
      }

      setPlayerStatus('ready');
      // resume playing if store indicates playback
      if (isPlaying) {
        video.play().catch(() => {});
        mapVideoRef.current?.play?.().catch?.(() => {});
      }
    };
    const onPlay = () => { setIsPlaying(true); setPlayerStatus('playing'); };
    const onPause = () => { setIsPlaying(false); setPlayerStatus('paused'); };
    const onTime = () => { setCurrentTime(video.currentTime); syncSlave(); };

    const onPlaySync = () => {
      try {
        if (mapVideoRef.current) mapVideoRef.current.play().catch(() => {});
      } catch {}
    };
    const onPauseSync = () => {
      try {
        if (mapVideoRef.current) mapVideoRef.current.pause();
      } catch {}
    };

    video.addEventListener('loadedmetadata', onLoaded);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    video.addEventListener('timeupdate', onTime);
    video.addEventListener('play', onPlaySync);
    video.addEventListener('pause', onPauseSync);

    // If we already have frame data, mark backend as online
    if (frameData && frameData.length > 0) setBackendStatus('online');

    return () => {
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
      video.removeEventListener('timeupdate', onTime);
      video.removeEventListener('play', onPlaySync);
      video.removeEventListener('pause', onPauseSync);
    };
  }, [frameData, setCurrentTime, setIsPlaying, currentTime, isPlaying]);

  useEffect(() => {
    const mv = mapVideoRef.current;
    if (!mv) return;
    const onLoaded = () => {
      try {
        if ((mv.readyState || 0) >= 1) mv.currentTime = currentTime || 0;
      } catch {}
    };
    mv.addEventListener('loadedmetadata', onLoaded);
    return () => {
      mv.removeEventListener('loadedmetadata', onLoaded);
    };
  }, [currentTime, mapUrl]);

  // --- UI and Display Logic (mostly unchanged) ---
  
  useEffect(() => {
    if (frameData.length === 0) { setCurrentFrameData(null); return; }
    const suitableFrames = frameData.filter(f => f.frame_id <= currentTime);
    if (suitableFrames.length > 0) {
      const closestFrame = suitableFrames.reduce((prev, curr) => (Math.abs(curr.frame_id - currentTime) < Math.abs(prev.frame_id - currentTime) ? curr : prev));
      setCurrentFrameData(closestFrame);
      setCurrentFrame(closestFrame.frame_id);
    }
  }, [currentTime, frameData, setCurrentFrame]);

  const drawOverlay = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !currentFrameData) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    currentFrameData.detections.forEach((detection) => {
      const x = detection.x * canvas.width;
      const y = detection.y * canvas.height;
      const width = detection.width * canvas.width;
      const height = detection.height * canvas.height;
      const colors = DETECTION_COLORS[detection.class_name] || DETECTION_COLORS.player;
      const labelText = `${colors.label} ${(detection.confidence * 100).toFixed(1)}%`;
      ctx.strokeStyle = colors.stroke;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);
      ctx.fillStyle = colors.fill;
      ctx.font = "14px monospace";
      const textMetrics = ctx.measureText(labelText);
      ctx.fillRect(x, y - 20, textMetrics.width + 8, 20);
      ctx.fillStyle = "#000";
      ctx.fillText(labelText, x + 4, y - 6);
    });
  }, [currentFrameData]);

  useEffect(() => { drawOverlay(); }, [drawOverlay]);
  
  const togglePlay = () => {
    if (playerStatus === 'buffering') return;
    const video = videoRef.current;
    if (video) video.paused ? video.play() : video.pause();
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const video = videoRef.current;
    const t = parseFloat(e.target.value);
    if (video) video.currentTime = t;
    if (mapVideoRef.current) {
      try { mapVideoRef.current.currentTime = t; } catch {}
    }
  };
  
  const handleClose = () => { selectVideo(null); };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`;

  if (!debugUrl) return null;

  return (
    <div className="max-w-6xl mx-auto">
      <canvas ref={frameCanvasRef} className="hidden" />
      <div>
        <Card className="overflow-hidden bg-black/5 dark:bg-black/20">
          <div className={mapUrl ? "relative grid grid-cols-1 lg:grid-cols-2 gap-2 p-2 bg-black" : "relative aspect-video bg-black"}>
            <div className={mapUrl ? "relative aspect-video bg-black" : ""}>
              <video ref={videoRef} src={debugUrl} className={mapUrl ? "absolute inset-0 w-full h-full object-contain" : "absolute inset-0 w-full h-full object-contain"} controls={false} muted crossOrigin="anonymous" />
              <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none" style={{ imageRendering: "pixelated" }} />
            </div>

            {mapUrl && (
              <div className="relative aspect-video bg-black">
                <video ref={mapVideoRef} src={mapUrl} className="absolute inset-0 w-full h-full object-contain" controls={false} muted crossOrigin="anonymous" />
              </div>
            )}
              
              {playerStatus === 'buffering' && backendStatus === 'online' && (
                <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center z-20">
                  <Loader2 className="h-8 w-8 animate-spin text-white mb-4" />
                  <p className="text-white font-medium">Buffering 100 frames...</p>
                </div>
              )}

              <div className="absolute top-4 right-4 z-10"> <Button variant="destructive" size="icon" onClick={handleClose} className="bg-black/50 hover:bg-black/70"> <X className="h-4 w-4" /> </Button> </div>
              
              {(isProcessingFrame || playerStatus === 'buffering') && (
                <div className="absolute bottom-4 left-4 z-10 bg-black/50 text-white px-3 py-1 rounded-md flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>{playerStatus === 'buffering' ? 'Buffering...' : 'Processing...'}</span>
                </div>
              )}
          </div>
            <div className="p-4 bg-card border-t">
              <div className="flex items-center gap-4 mb-4">
                <Button variant="brand" size="icon" onClick={togglePlay} disabled={playerStatus === 'buffering'}>
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                </Button>
                <div className="flex-1"> <input type="range" min="0" max={duration || 0} step="0.1" value={currentTime} onChange={handleSeek} className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-brand" /> </div>
                <div className="text-sm text-muted-foreground min-w-[100px] text-right"> {formatTime(currentTime)} / {formatTime(duration)} </div>
                <Button variant="outline" size="icon" onClick={() => { if (videoRef.current) videoRef.current.currentTime = 0; if (mapVideoRef.current) { try { mapVideoRef.current.currentTime = 0; } catch {} } }}> <RotateCcw className="h-4 w-4" /> </Button>
              </div>
            </div>
          </Card>

        <Card className="mt-4">
          <div className="p-4 border-b">
            <h3 className="font-semibold text-lg">Frame Data</h3>
            <p className="text-sm text-muted-foreground">Pre-processed model outputs</p>
          </div>
          <div className="p-4 overflow-auto max-h-72 debug-panel">
            {currentFrameData ? (
              <pre className="text-xs bg-muted p-4 rounded-lg overflow-auto font-mono">{JSON.stringify(currentFrameData, null, 2)}</pre>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                {backendStatus === 'online' ? "Play video to view frame data" : "No debug data available"}
              </div>
            )}
          </div>
          <div className="p-4 border-t text-sm text-muted-foreground space-y-2">
            <div>Total frames processed: {frameData.length}</div>
            <div>Current frame displayed: {currentFrame > 0 ? currentFrame.toFixed(3) : '0.000'}</div>
            <div className="flex items-center gap-2">
              <Server className={`h-4 w-4 ${backendStatus === 'online' ? 'text-green-500' : 'text-red-500'}`} />
              <span>Backend:</span>
              <span className={`capitalize font-medium ${backendStatus === 'online' ? 'text-green-500' : 'text-red-500'}`}> {backendStatus} </span>
            </div>
            <div className="flex items-center gap-2">
              <Framer className="h-4 w-4 text-brand" />
              <span>Processing FPS:</span>
              <span className="font-medium text-primary">{backendFps.toFixed(2)}</span>
            </div>
            {videoRef.current && <div className="mt-1">Playback Rate: {videoRef.current.playbackRate.toFixed(2)}x</div>}
          </div>
        </Card>
      </div>
    </div>
  );
}
