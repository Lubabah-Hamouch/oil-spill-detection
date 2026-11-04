export type BackendStatus = "checking" | "waking" | "ready";

const MESSAGES: Record<Exclude<BackendStatus, "ready">, string> = {
  checking: "Connecting to the analysis server…",
  waking: "Waking up the server — this can take up to a minute on first load.",
};

export default function ServerStatusBanner({ status }: { status: BackendStatus }) {
  if (status === "ready") return null;

  return (
    <div className="border-b border-line/80 bg-surface">
      <div
        role="status"
        aria-live="polite"
        className="mx-auto flex max-w-5xl items-center gap-3 px-5 py-3 text-sm text-ink-muted sm:px-8"
      >
        <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-accent" />
        {MESSAGES[status]}
      </div>
      <div className="h-0.5 animate-shimmer bg-[linear-gradient(90deg,#1A212B_25%,#2DD4BF_50%,#1A212B_75%)] bg-[length:200%_100%]" />
    </div>
  );
}
