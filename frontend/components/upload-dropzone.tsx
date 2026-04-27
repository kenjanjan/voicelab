"use client";
import { useRef, useState } from "react";
import { uploadAudio } from "@/lib/api";
import { Button } from "@/components/ui/button";

export function UploadDropzone({ creatorId, onDone }: { creatorId: string; onDone: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);

  async function handle(files: FileList | null) {
    if (!files || files.length === 0) return;
    setBusy(true);
    try {
      await uploadAudio(creatorId, Array.from(files));
      onDone();
    } catch (e) {
      alert(`Upload failed: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files); }}
      className={`rounded-lg border-2 border-dashed p-8 text-center transition
        ${drag ? "border-[var(--color-accent)] bg-[var(--color-panel)]" : "border-[var(--color-border)]"}`}
    >
      <p className="text-sm text-[var(--color-muted)] mb-3">
        Drop .wav / .mp3 / .m4a here, or
      </p>
      <input
        ref={inputRef} type="file" multiple accept="audio/*" hidden
        onChange={(e) => handle(e.target.files)}
      />
      <Button onClick={() => inputRef.current?.click()} disabled={busy}>
        {busy ? "Uploading…" : "Choose files"}
      </Button>
    </div>
  );
}
