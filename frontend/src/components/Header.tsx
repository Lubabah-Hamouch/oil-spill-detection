function LogoMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-9 w-9" aria-hidden="true">
      <rect width="32" height="32" rx="9" className="fill-surface" />
      <rect x="0.5" y="0.5" width="31" height="31" rx="8.5" className="fill-none stroke-line" />
      <circle cx="16" cy="16" r="9" className="fill-none stroke-accent" strokeWidth="1.75" />
      <circle cx="16" cy="16" r="5" className="fill-none stroke-accent/40" strokeWidth="1.25" />
      <circle cx="16" cy="16" r="2.5" className="fill-spill" />
    </svg>
  );
}

export default function Header() {
  return (
    <header className="border-b border-line/80">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-5 py-4 sm:px-8">
        <LogoMark />
        <div className="leading-tight">
          <h1 className="text-[15px] font-semibold tracking-tight">Oil Spill Detection</h1>
          <p className="text-xs text-ink-muted">Satellite Image Analysis</p>
        </div>
      </div>
    </header>
  );
}
