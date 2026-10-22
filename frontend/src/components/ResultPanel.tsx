import type { ConfidenceLabel, PredictResult } from "../api/client";

export type AnalysisState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; result: PredictResult };

const LABELS: Record<ConfidenceLabel, { title: string; detail: string; tone: "spill" | "clear" }> = {
  NO_SPILL: { title: "No spill detected", detail: "The model found no evidence of oil in this image.", tone: "clear" },
  LOW_CONFIDENCE_SPILL: {
    title: "Possible spill",
    detail: "Low-confidence detection. Manual verification recommended.",
    tone: "spill",
  },
  HIGH_CONFIDENCE_SPILL: {
    title: "Oil spill detected",
    detail: "Strong visual evidence of an oil spill.",
    tone: "spill",
  },
};

const TONE = {
  spill: { text: "text-spill", bar: "bg-spill", ring: "border-spill/40 bg-spill/10" },
  clear: { text: "text-clear", bar: "bg-clear", ring: "border-clear/40 bg-clear/10" },
} as const;

function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`animate-shimmer rounded-lg bg-[linear-gradient(90deg,#1A212B_25%,#232B36_50%,#1A212B_75%)] bg-[length:200%_100%] ${className}`}
    />
  );
}

function Loading() {
  return (
    <div className="space-y-5" role="status" aria-live="polite">
      <div className="flex items-center gap-2 text-sm text-ink-muted">
        <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
        Analyzing image…
      </div>
      <Skeleton className="h-7 w-48" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

function Empty() {
  return (
    <div className="flex h-full min-h-[14rem] flex-col items-center justify-center gap-2 text-center">
      <div className="h-2 w-2 rounded-full bg-line" />
      <p className="text-sm text-ink-muted">Results will appear here after analysis.</p>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-xl border border-spill/30 bg-spill/5 p-4">
      <p className="text-sm font-semibold text-spill">Analysis failed</p>
      <p className="mt-1 text-sm text-ink-muted">{message}</p>
    </div>
  );
}

function Success({ result }: { result: PredictResult }) {
  const meta = LABELS[result.confidence_label] ?? LABELS.NO_SPILL;
  const tone = TONE[meta.tone];
  const pct = result.probability * 100;

  return (
    <div className="animate-rise space-y-6">
      <div>
        <span
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${tone.ring} ${tone.text}`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${tone.bar}`} />
          {result.confidence_label.replace(/_/g, " ")}
        </span>
        <h2 className={`mt-3 text-2xl font-semibold tracking-tight ${tone.text}`}>{meta.title}</h2>
        <p className="mt-1 text-sm text-ink-muted">{meta.detail}</p>
      </div>

      <div>
        <div className="flex items-baseline justify-between">
          <span className="eyebrow">Spill probability</span>
          <span className="font-mono text-xl font-semibold tabular-nums">{pct.toFixed(1)}%</span>
        </div>
        <div
          className="mt-2 h-2 overflow-hidden rounded-full bg-elevated"
          role="img"
          aria-label={`Spill probability ${pct.toFixed(1)} percent`}
        >
          <div
            className={`h-full rounded-full ${tone.bar} transition-[width] duration-700`}
            style={{ width: `${Math.max(pct, 1)}%` }}
          />
        </div>
      </div>

      {result.spill_detected && result.overlayUrl && (
        <div className="space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="eyebrow">Spill localization</span>
            {result.area_percentage !== null && (
              <span className="text-sm text-ink-muted">
                <span className="font-mono font-semibold tabular-nums text-spill">
                  {result.area_percentage.toFixed(1)}%
                </span>{" "}
                of image area
              </span>
            )}
          </div>
          <img
            src={result.overlayUrl}
            alt="Segmentation overlay with detected spill highlighted"
            className="w-full rounded-xl border border-line bg-canvas"
          />
        </div>
      )}

      {result.spill_detected && !result.overlayUrl && (
        <p className="rounded-lg border border-line bg-elevated/50 px-3 py-2 text-xs text-ink-muted">
          Localization unavailable: {result.segmentation_status}.
        </p>
      )}
    </div>
  );
}

export default function ResultPanel({ state }: { state: AnalysisState }) {
  return (
    <section className="panel p-6" aria-label="Analysis results">
      <div className="eyebrow mb-4">Analysis</div>
      {state.status === "idle" && <Empty />}
      {state.status === "loading" && <Loading />}
      {state.status === "error" && <ErrorState message={state.message} />}
      {state.status === "success" && <Success result={state.result} />}
    </section>
  );
}
