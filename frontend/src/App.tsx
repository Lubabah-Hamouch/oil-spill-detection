import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, checkHealth, predict } from "./api/client";
import Footer from "./components/Footer";
import Header from "./components/Header";
import ResultPanel, { type AnalysisState } from "./components/ResultPanel";
import ServerStatusBanner, { type BackendStatus } from "./components/ServerStatusBanner";
import UploadZone from "./components/UploadZone";

// Delay between health checks while the backend is asleep or starting up.
const HEALTH_POLL_INTERVAL_MS = 3500;

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: "idle" });
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const abortRef = useRef<AbortController | null>(null);

  // Poll /health until the backend answers. The page renders regardless; only analysis waits on it.
  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;

    async function poll() {
      const ok = await checkHealth(controller.signal);
      if (controller.signal.aborted) return;
      if (ok) {
        setBackendStatus("ready");
        return;
      }
      setBackendStatus("waking");
      timer = window.setTimeout(poll, HEALTH_POLL_INTERVAL_MS);
    }

    poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, []);

  // Create/revoke the object URL for the preview.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleSelect = useCallback((next: File) => {
    abortRef.current?.abort();
    setFile(next);
    setAnalysis({ status: "idle" });
  }, []);

  const handleReject = useCallback((message: string) => {
    setAnalysis({ status: "error", message });
  }, []);

  async function handleAnalyze() {
    if (!file) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setAnalysis({ status: "loading" });
    try {
      const result = await predict(file, controller.signal);
      setAnalysis({ status: "success", result });
    } catch (err) {
      if (controller.signal.aborted) return;
      const message = err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
      setAnalysis({ status: "error", message });
    }
  }

  const loading = analysis.status === "loading";
  const backendReady = backendStatus === "ready";

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <ServerStatusBanner status={backendStatus} />

      <main className="mx-auto w-full max-w-5xl flex-1 px-5 py-10 sm:px-8">
        <div className="mb-8 max-w-xl">
          <p className="eyebrow text-accent">Monitoring console</p>
          <h2 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Detect and localize oil spills.</h2>
          <p className="mt-3 text-sm leading-relaxed text-ink-muted">
            Upload an aerial or satellite image patch. The model classifies it and, if a spill is likely, outlines where.
          </p>
        </div>

        <div className="grid items-start gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            <UploadZone
              previewUrl={previewUrl}
              fileName={file?.name ?? null}
              disabled={loading}
              onSelect={handleSelect}
              onReject={handleReject}
            />
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={!file || loading || !backendReady}
              className="w-full rounded-xl bg-accent px-5 py-3 text-sm font-semibold text-canvas transition hover:brightness-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60 focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:cursor-not-allowed disabled:bg-elevated disabled:text-ink-muted"
            >
              {loading ? "Analyzing…" : "Analyze image"}
            </button>
          </div>

          <ResultPanel state={analysis} />
        </div>
      </main>

      <Footer />
    </div>
  );
}
