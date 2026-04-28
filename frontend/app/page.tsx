"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api, SystemStatus } from "@/lib/api";

export default function Home() {
  const { data, mutate, error, isLoading } = useSWR<SystemStatus>(
    "/api/system/status", api.systemStatus,
    { refreshInterval: (d) => (d?.setup.status === "running" ? 1500 : 30000) },
  );
  const [busy, setBusy] = useState(false);

  async function startSetup() {
    setBusy(true);
    try {
      await api.triggerSetup();
      await mutate();
    } finally {
      setBusy(false);
    }
  }

  if (error) {
    return (
      <Card className="border-red-700/50">
        <CardTitle>Backend unreachable</CardTitle>
        <p className="text-sm text-[var(--color-muted)]">
          The frontend cannot reach the backend at <code>http://localhost:8000</code>.
          Start it with <code>uvicorn app.main:app --reload --port 8000</code> in the
          backend folder.
        </p>
      </Card>
    );
  }
  if (isLoading || !data) {
    return <p className="text-sm text-[var(--color-muted)]">Loading status…</p>;
  }

  const setupRunning = data.setup.status === "running";
  const setupFailed = data.setup.status === "failed";
  const ready = data.models.cosyvoice_repo && data.models.cosyvoice_weights;

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">VoiceLab</h1>
        <p className="text-[var(--color-muted)] mt-2">
          Onboard a creator, build a dataset, fine-tune LoRA, generate emotional voice messages — all from this UI.
        </p>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardTitle>Hardware</CardTitle>
          <Row label="Backend" value={data.backend_ok ? "online" : "offline"} ok={data.backend_ok} />
          <Row
            label="Compute"
            value={data.gpu.available ? `${data.gpu.name}${data.gpu.vram_gb ? ` · ${data.gpu.vram_gb} GB` : ""}` : "CPU only — slow"}
            ok={data.gpu.available && data.gpu.kind === "cuda"}
            warn={data.gpu.available && data.gpu.kind !== "cuda"}
          />
        </Card>

        <Card>
          <CardTitle>Models</CardTitle>
          <Row label="CosyVoice repo" value={data.models.cosyvoice_repo ? "present" : "missing"} ok={data.models.cosyvoice_repo} />
          <Row label="CosyVoice2-0.5B weights" value={data.models.cosyvoice_weights ? "present" : "missing"} ok={data.models.cosyvoice_weights} />

          {!ready && (
            <div className="mt-4">
              <Button onClick={startSetup} disabled={busy || setupRunning}>
                {setupRunning ? "Downloading…" : "Download CosyVoice 2 (~2GB)"}
              </Button>
              {setupRunning && (
                <div className="mt-3">
                  <div className="text-xs text-[var(--color-muted)] mb-1">
                    {data.setup.step ?? "starting"} · {Math.round(data.setup.progress * 100)}%
                  </div>
                  <div className="h-1.5 bg-[var(--color-bg)] rounded">
                    <div className="h-1.5 bg-[var(--color-accent)] rounded transition-all"
                         style={{ width: `${Math.round(data.setup.progress * 100)}%` }} />
                  </div>
                </div>
              )}
              {setupFailed && (
                <p className="text-xs text-red-400 mt-2">
                  Setup failed — see log below.
                </p>
              )}
            </div>
          )}

          {data.setup.log && (
            <pre className="mt-3 text-[10px] text-[var(--color-muted)] max-h-40 overflow-auto whitespace-pre-wrap">
              {data.setup.log.split("\n").slice(-12).join("\n")}
            </pre>
          )}
        </Card>

        <Card>
          <CardTitle>Workspace</CardTitle>
          <Row label="Creators" value={String(data.counts.creators)} />
          <Row label="Clips" value={String(data.counts.clips)} />
          <Row label="Reference clips" value={String(data.counts.reference_clips)} />
          <p className="text-xs text-[var(--color-muted)] mt-3">
            Storage: <span className="font-mono">{data.storage.data_dir}</span>
          </p>
        </Card>

        <Card>
          <CardTitle>Next step</CardTitle>
          {!ready ? (
            <p className="text-sm text-[var(--color-muted)]">
              Download CosyVoice 2 to enable training and synthesis.
            </p>
          ) : data.counts.creators === 0 ? (
            <>
              <p className="text-sm text-[var(--color-muted)] mb-3">Create your first creator.</p>
              <Link href="/creators" className="bg-[var(--color-accent)] text-white px-4 py-2 rounded-md text-sm font-medium">
                Go to Creators →
              </Link>
            </>
          ) : data.counts.reference_clips === 0 ? (
            <>
              <p className="text-sm text-[var(--color-muted)] mb-3">
                Upload audio, preprocess, and mark reference clips per emotion.
              </p>
              <Link href="/creators" className="bg-[var(--color-accent)] text-white px-4 py-2 rounded-md text-sm font-medium">
                Open creators →
              </Link>
            </>
          ) : (
            <>
              <p className="text-sm text-[var(--color-muted)] mb-3">
                You have references — start a LoRA run or synthesize directly.
              </p>
              <Link href="/creators" className="bg-[var(--color-accent)] text-white px-4 py-2 rounded-md text-sm font-medium">
                Open creators →
              </Link>
            </>
          )}
        </Card>
      </div>
    </div>
  );
}

function Row({ label, value, ok, warn }: { label: string; value: string; ok?: boolean; warn?: boolean }) {
  const dot = ok ? "bg-emerald-400" : warn ? "bg-amber-400" : ok === false ? "bg-red-400" : "bg-zinc-500";
  return (
    <div className="flex items-center justify-between text-sm py-1.5 border-b border-[var(--color-border)] last:border-b-0">
      <span className="text-[var(--color-muted)]">{label}</span>
      <span className="flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${dot}`} />
        <span>{value}</span>
      </span>
    </div>
  );
}
