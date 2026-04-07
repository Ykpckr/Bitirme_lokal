"use client";

import { useEffect, useRef, useState } from "react";
import { useAppStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Loader2, Upload } from "lucide-react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

interface VideoFile {
  path: string;
  name: string;
  url?: string;
  kind?: string;
}

function _extractRunIdFromFilename(name: string): { prefix: string; runId: string } | null {
  const base = String(name || "");
  const lower = base.toLowerCase();
  if (!lower.includes(".")) return null;
  const stem = base.slice(0, base.lastIndexOf("."));
  const i = stem.indexOf("_");
  if (i <= 0) return null;
  const prefix = stem.slice(0, i);
  const rest = stem.slice(i + 1);
  if (!rest) return null;
  return { prefix, runId: rest };
}

export function VideoSelector() {
  const selectVideo = useAppStore((state) => state.selectVideo);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [minutes, setMinutes] = useState<string>("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [processedVideos, setProcessedVideos] = useState<VideoFile[]>([]);
  const [listError, setListError] = useState<string | null>(null);

  async function refreshProcessedVideos() {
    try {
      setListError(null);
      const resp = await fetch(`${BACKEND_URL}/api/uploads`, { cache: "no-store" });
      if (!resp.ok) throw new Error(await resp.text());
      const data = (await resp.json()) as VideoFile[];
      setProcessedVideos(Array.isArray(data) ? data : []);
    } catch (e: any) {
      setListError(e?.message || String(e));
    }
  }

  useEffect(() => {
    refreshProcessedVideos();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function isLikelyVideoFile(f: File): boolean {
    const name = (f?.name || "").toLowerCase();
    const ext = name.includes(".") ? name.slice(name.lastIndexOf(".")) : "";
    if (f?.type && f.type.startsWith("video/")) return true;
    return [".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"].includes(ext);
  }

  function pickFile() {
    fileInputRef.current?.click();
  }

  function handleFile(f: File | null) {
    if (!f) {
      setSelectedFile(null);
      return;
    }
    if (!isLikelyVideoFile(f)) {
      setStatusMessage("Lütfen bir video dosyası seçin (.mkv dahil).");
      return;
    }
    setSelectedFile(f);
    setStatusMessage(null);
  }

  useEffect(() => {
    if (!selectedFile) {
      if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
      setLocalPreviewUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(selectedFile);
    setLocalPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return nextUrl;
    });
    return () => {
      try {
        URL.revokeObjectURL(nextUrl);
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedFile]);

  async function uploadAndRunPipeline() {
    if (!selectedFile) {
      setStatusMessage("Lütfen bir video dosyası seçin.");
      return;
    }

    setBusy(true);
    setStatusMessage("Video yükleniyor...");

    try {
      const fd = new FormData();
      fd.append("file", selectedFile);

      const uploadResp = await fetch(`${BACKEND_URL}/api/upload_video`, {
        method: "POST",
        body: fd,
      });
      if (!uploadResp.ok) {
        throw new Error(await uploadResp.text());
      }
      const uploadData = await uploadResp.json();
      const uploadedPath = uploadData.path as string;

      setStatusMessage("Pipeline çalışıyor (tracking + action spotting)...");

      const minVal = minutes.trim() === "" ? null : parseFloat(minutes);
      const payload: any = {
        path: uploadedPath,
        start_seconds: 0.0,
        run_tracking: true,
        run_action_spotting: true,
      };
      if (minVal !== null && !isNaN(minVal) && minVal > 0) {
        payload.minutes = minVal;
      }

      const runResp = await fetch(`${BACKEND_URL}/api/run_full_pipeline_async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!runResp.ok) {
        throw new Error(await runResp.text());
      }
      const runData = await runResp.json();
      const jobId = runData.job_id as string;

      setStatusMessage(`Pipeline başladı. Job: ${jobId}`);

      await new Promise<void>((resolve, reject) => {
        const es = new EventSource(`${BACKEND_URL}/api/pipeline_progress?job_id=${encodeURIComponent(jobId)}`);

        const cleanup = () => {
          try { es.close(); } catch {}
        };

        es.onmessage = (ev) => {
          try {
            const data = JSON.parse(ev.data);

            if (data?.type === "result" && data?.result) {
              const overlayUrl = (data.result.debug_video_url as string) || (data.result.overlay_video_url as string);
              const overlayPath = data.result.overlay_video_path as string;
              const productUrl = (data.result.product_video_url as string) || (data.result.commentary_video_url as string) || overlayUrl;
              const productPath = (data.result.product_video_path as string) || (data.result.commentary_video_path as string) || overlayPath;
              const mapUrl = (data.result.calibration_map_video_url as string) || null;
              selectVideo({
                normalUrl: productUrl,
                debugUrl: overlayUrl,
                mapUrl,
                path: productPath || overlayPath || uploadedPath,
                name: (productPath || overlayPath || "video.mp4").split("/").pop() || "video.mp4",
              });
              setStatusMessage("Tamamlandı.");
              refreshProcessedVideos();
              cleanup();
              resolve();
              return;
            }

            if (data?.type === "error") {
              setStatusMessage(`Hata: ${data?.error || "Bilinmeyen hata"}`);
              cleanup();
              reject(new Error(data?.error || "pipeline error"));
              return;
            }

            if (data?.status) {
              const cur = Number(data.current || 0);
              const total = Number(data.total || 0);
              const pct = total > 0 ? Math.floor((cur * 100) / total) : null;
              const msg = String(data.message || "");
              const stage = String(data.stage || data.status);
              setStatusMessage(pct !== null ? `[${stage}] ${pct}%  ${msg}` : `[${stage}] ${msg}`);
            }
          } catch (e) {
            // ignore parse errors
          }
        };

        es.onerror = () => {
          setStatusMessage("Progress bağlantısı koptu.");
          cleanup();
          reject(new Error("SSE error"));
        };
      });
    } catch (e: any) {
      console.error(e);
      setStatusMessage(`İşlem başarısız: ${e?.message || e}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-3">
      <Card className="h-fit">
        <CardHeader>
          <CardTitle className="text-base">Overlay Videoları</CardTitle>
          <CardDescription>Mevcut overlay çıktıları</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-sm font-medium">Liste</div>
            <Button type="button" variant="secondary" size="sm" onClick={refreshProcessedVideos} disabled={busy}>
              Yenile
            </Button>
          </div>

          {listError ? (
            <div className="text-sm text-destructive">Liste okunamadı: {listError}</div>
          ) : processedVideos.filter((v) => (v.kind || "").toLowerCase() === "overlay" || v.name.toLowerCase().startsWith("overlay_")).length === 0 ? (
            <div className="text-sm text-muted-foreground">Henüz overlay video yok.</div>
          ) : (
            <div className="grid grid-cols-1 gap-2">
              {processedVideos
                .filter((v) => (v.kind || "").toLowerCase() === "overlay" || v.name.toLowerCase().startsWith("overlay_"))
                .slice()
                .reverse()
                .map((v) => (
                  <div key={v.path} className="flex items-center justify-between gap-2 rounded-md border p-2">
                    <div className="min-w-0">
                      <div className="text-xs font-medium truncate">{v.name}</div>
                    </div>
                    <Button
                      type="button"
                      size="sm"
                      onClick={() => {
                        const overlayUrl = v.url || `${BACKEND_URL}/api/video?path=${encodeURIComponent(v.path)}`;
                        let productUrl = overlayUrl;
                        let productPath = v.path;

                        // If this looks like overlay_<runid>.mp4, try to locate a matching product_<runid>_commentary.mp4
                        // so that Debug Mode toggles between clean+audio vs boxed overlay.
                        const parsed = _extractRunIdFromFilename(v.name);
                        if (parsed && parsed.prefix.toLowerCase() === "overlay") {
                          const runId = parsed.runId;
                          const candidate = processedVideos.find((x) => {
                            const ln = (x.name || "").toLowerCase();
                            return ln.startsWith(`product_${runId.toLowerCase()}_`) || ln.startsWith(`product_${runId.toLowerCase()}.`);
                          });
                          if (candidate) {
                            productPath = candidate.path;
                            productUrl = candidate.url || `${BACKEND_URL}/api/video?path=${encodeURIComponent(candidate.path)}`;
                          }
                        }

                        let mapUrl: string | null = null;
                        if (parsed && parsed.prefix.toLowerCase() === "overlay") {
                          const runId = parsed.runId;
                          const mapCandidate = processedVideos.find((x) => (x.name || "").toLowerCase().startsWith(`map_${runId.toLowerCase()}.`));
                          if (mapCandidate) {
                            mapUrl = mapCandidate.url || `${BACKEND_URL}/api/video?path=${encodeURIComponent(mapCandidate.path)}`;
                          }
                        }

                        selectVideo({
                          normalUrl: productUrl,
                          debugUrl: overlayUrl,
                          mapUrl,
                          path: productPath,
                          name: (productPath || v.path).split("/").pop() || v.name,
                        });
                        setStatusMessage("Overlay seçildi.");
                      }}
                    >
                      Oynat
                    </Button>
                  </div>
                ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-3xl mb-2">Video Yükle</CardTitle>
          <CardDescription className="text-lg">Herhangi bir videoyu yükle, pipeline çalışsın ve çıktı videoyu izle.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <div className="text-sm font-medium">Video Alanı (1920×1080)</div>

            <div className="w-full rounded-md border overflow-hidden">
              <div
                className={`w-full aspect-video relative ${dragActive ? "bg-muted/60" : "bg-muted/30"}`}
                onDragEnterCapture={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setDragActive(true);
                }}
                onDragOverCapture={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setDragActive(true);
                  try {
                    e.dataTransfer.dropEffect = "copy";
                  } catch {
                    // ignore
                  }
                }}
                onDragLeaveCapture={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setDragActive(false);
                }}
                onDropCapture={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setDragActive(false);
                  const f = e.dataTransfer?.files?.[0] || null;
                  handleFile(f);
                }}
                onClick={() => {
                  // Only auto-open picker when empty; when preview exists,
                  // clicking should not break video controls.
                  if (!localPreviewUrl) pickFile();
                }}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (!localPreviewUrl && (e.key === "Enter" || e.key === " ")) pickFile();
                }}
              >
                {localPreviewUrl ? (
                  <>
                    <video
                      src={localPreviewUrl}
                      controls
                      className="absolute inset-0 w-full h-full object-contain bg-black"
                    />
                    <div className="absolute top-3 left-3 z-10">
                      <Button
                        type="button"
                        variant="secondary"
                        size="sm"
                        className="bg-black/50 hover:bg-black/70 text-white"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          pickFile();
                        }}
                      >
                        Değiştir
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6">
                    <Upload className="h-8 w-8 text-muted-foreground" />
                    <div className="mt-3 text-sm text-muted-foreground">
                      Videoyu buraya sürükle-bırak veya tıkla. (.mkv dahil)
                    </div>
                  </div>
                )}
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,.mkv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0] || null;
                handleFile(f);
              }}
            />

            <div className="text-xs text-muted-foreground">Yükleme sonrası video otomatik pipeline’a girer.</div>
          </div>

          <div className="flex items-center gap-2">
            <div className="text-sm font-medium">Dakika (opsiyonel)</div>
            <Input
              value={minutes}
              onChange={(e) => setMinutes(e.target.value)}
              type="number"
              min="0.05"
              step="0.1"
              placeholder="0.5"
              className="w-32"
              disabled={busy}
            />
            <div className="text-xs text-muted-foreground">Boş bırakırsan tüm video.</div>
          </div>

          {statusMessage ? <div className="text-sm">{statusMessage}</div> : null}

          <div className="flex justify-end">
            <Button onClick={uploadAndRunPipeline} disabled={busy || !selectedFile}>
              {busy ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  İşleniyor...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Yükle ve İşle
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
