export default function Footer() {
  return (
    <footer className="border-t border-line/80">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-2 px-5 py-5 text-xs text-ink-muted sm:px-8">
        <span>Decision-support demo. Verify detections before acting on them.</span>
        <a
          href="https://github.com/"
          target="_blank"
          rel="noreferrer"
          className="transition-colors hover:text-accent"
        >
          GitHub
        </a>
      </div>
    </footer>
  );
}
