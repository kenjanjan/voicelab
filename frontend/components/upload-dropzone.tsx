"use client";
import { useRef, useState } from "react";
import { uploadAudio } from "@/lib/api";
import { Button } from "@/components/ui/button";

type ItemStatus = "queued" | "uploading" | "done" | "error" | "aborted";
type Item = {
  file: File;
  loaded: number;
  total: number;
  status: ItemStatus;
  error?: string;
  abort?: AbortController;
};

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export function UploadDropzone({ creatorId, onDone }: { creatorId: string; onDone: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);

  function patch(idx: number, fields: Partial<Item>) {
    setItems((prev) => prev.map((it, i) => (i === idx ? { ...it, ...fields } : it)));
  }

  async function uploadOne(idx: number, file: File) {
    const ac = new AbortController();
    patch(idx, { status: "uploading", abort: ac });
    try {
      await uploadAudio(
        creatorId, file,
        (loaded, total) => patch(idx, { loaded, total }),
        ac.signal,
      );
      patch(idx, { status: "done", loaded: file.size, total: file.size });
    } catch (e) {
      const err = e as Error;
      const aborted = err.name === "AbortError";
      patch(idx, { status: aborted ? "aborted" : "error", error: aborted ? undefined : err.message });
    }
  }

  async function add(picked: FileList | null) {
    if (!picked || picked.length === 0) return;
    const newItems: Item[] = Array.from(picked).map((f) => ({
      file: f, loaded: 0, total: f.size, status: "queued",
    }));
    const startIdx = items.length;
    setItems((prev) => [...prev, ...newItems]);
    setBusy(true);
    for (let i = 0; i < newItems.length; i++) {
      await uploadOne(startIdx + i, newItems[i].file);
    }
    setBusy(false);
    onDone();
  }

  function clearDone() {
    setItems((prev) => prev.filter((it) => it.status !== "done"));
  }

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
        className={`rounded-lg border-2 border-dashed p-8 text-center transition
          ${drag ? "border-[var(--color-accent)] bg-[var(--color-panel)]" : "border-[var(--color-border)]"}`}
      >
        <p className="text-sm text-[var(--color-muted)] mb-3">
          Drop audio files here, or
        </p>
        <input
          ref={inputRef} type="file" multiple accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.aac"
          hidden onChange={(e) => add(e.target.files)}
        />
        <Button onClick={() => inputRef.current?.click()} disabled={busy}>
          Choose files
        </Button>
        <p className="text-xs text-[var(--color-muted)] mt-2">
          Multiple files supported. Each file uploads sequentially with live progress — large files (multi-GB) ok.
        </p>
      </div>

      {items.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs text-[var(--color-muted)]">
              {items.filter((i) => i.status === "done").length}/{items.length} uploaded
            </p>
            <button
              onClick={clearDone}
              className="text-xs text-[var(--color-muted)] hover:text-white"
            >
              Clear completed
            </button>
          </div>
          <ul className="space-y-1.5">
            {items.map((it, i) => {
              const pct = it.total > 0 ? Math.round((it.loaded / it.total) * 100) : 0;
              const barColor =
                it.status === "error" ? "bg-red-500" :
                it.status === "done" ? "bg-emerald-500" :
                it.status === "aborted" ? "bg-amber-500" :
                "bg-[var(--color-accent)]";
              return (
                <li key={i} className="text-xs">
                  <div className="flex items-center gap-3">
                    <span className="flex-1 truncate font-mono">{it.file.name}</span>
                    <span className="text-[var(--color-muted)] w-20 text-right">
                      {fmtBytes(it.file.size)}
                    </span>
                    <span className="w-12 text-right tabular-nums">
                      {it.status === "uploading" ? `${pct}%` :
                       it.status === "done" ? "✓" :
                       it.status === "error" ? "fail" :
                       it.status === "aborted" ? "abort" : "queued"}
                    </span>
                    {it.status === "uploading" && (
                      <button
                        onClick={() => it.abort?.abort()}
                        className="text-[var(--color-muted)] hover:text-red-400"
                        title="Cancel"
                      >
                        ×
                      </button>
                    )}
                  </div>
                  <div className="h-1 bg-[var(--color-bg)] rounded mt-1 overflow-hidden">
                    <div className={`h-1 ${barColor} transition-all`} style={{ width: `${pct}%` }} />
                  </div>
                  {it.error && (
                    <p className="text-red-400 mt-0.5 truncate" title={it.error}>{it.error}</p>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
