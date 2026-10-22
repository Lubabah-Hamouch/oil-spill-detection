import { useRef, useState, type DragEvent, type KeyboardEvent } from "react";

interface Props {
  previewUrl: string | null;
  fileName: string | null;
  disabled?: boolean;
  onSelect: (file: File) => void;
  onReject: (message: string) => void;
}

const MAX_BYTES = 20 * 1024 * 1024; // matches the backend limit

export default function UploadZone({ previewUrl, fileName, disabled, onSelect, onReject }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function accept(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) return onReject("That file isn't an image. Please choose a PNG or JPEG.");
    if (file.size > MAX_BYTES) return onReject("That image is larger than 20 MB. Please choose a smaller file.");
    onSelect(file);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (!disabled) accept(e.dataTransfer.files[0]);
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      inputRef.current?.click();
    }
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload a satellite image"
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={onKeyDown}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={[
        "group relative flex min-h-[18rem] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed text-center transition-colors",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60",
        dragging ? "border-accent bg-accent/5" : "border-line bg-surface hover:border-accent/50 hover:bg-elevated/60",
        disabled ? "pointer-events-none opacity-70" : "",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = ""; // allow re-selecting the same file
        }}
      />

      {previewUrl ? (
        <>
          <img
            src={previewUrl}
            alt="Selected satellite image"
            className="absolute inset-0 h-full w-full bg-canvas/60 object-contain"
          />
          <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 bg-gradient-to-t from-canvas/90 to-transparent px-4 pb-3 pt-10 text-left">
            <span className="truncate text-xs text-ink-muted">{fileName}</span>
            <span className="shrink-0 text-xs font-medium text-accent opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100">
              Replace image
            </span>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center gap-3 px-6">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-line bg-elevated text-accent">
            <svg
              viewBox="0 0 24 24"
              className="h-6 w-6"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
              <path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium">Drop a satellite image or click to upload</p>
            <p className="mt-1 text-xs text-ink-muted">PNG or JPEG, up to 20 MB</p>
          </div>
        </div>
      )}
    </div>
  );
}
