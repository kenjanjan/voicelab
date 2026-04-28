"use client";
import Link from "next/link";
import { use, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { UploadDropzone } from "@/components/upload-dropzone";
import { api, Clip, Creator, Job, SystemStatus } from "@/lib/api";

const EMOTIONS = ["casual", "flirty", "seductive", "excited", "playful_giggle", "soft_intimate"];

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function autoLoraSettings(
  nClips: number, totalSec: number, vramGb: number | null, gpuKind: string,
): { rank: number; epochs: number; lr: number; reason: string; warn?: string } {
  let rank: number;
  if (gpuKind !== "cuda") rank = 4;
  else if (vramGb == null) rank = 8;
  else if (vramGb <= 6) rank = 8;
  else if (vramGb <= 10) rank = 12;
  else rank = 16;

  let epochs: number;
  if (nClips === 0) epochs = 8;
  else if (nClips >= 4000) epochs = 1;
  else if (nClips >= 1500) epochs = 2;
  else if (nClips >= 600) epochs = 4;
  else if (nClips >= 250) epochs = 8;
  else if (nClips >= 100) epochs = 12;
  else epochs = 15;

  const lr = nClips < 50 ? 5e-5 : nClips > 1500 ? 1.5e-4 : 1e-4;

  const minutes = Math.round(totalSec / 60);
  const totalSteps = nClips * epochs;
  const reason =
    `${nClips} clips · ~${minutes} min audio · ` +
    (gpuKind === "cuda" ? `${vramGb ?? "?"} GB VRAM`
      : gpuKind === "mps" ? "Apple MPS" : "CPU only") +
    ` → ${epochs} epochs × rank ${rank} ≈ ${totalSteps.toLocaleString()} steps, lr ${lr.toExponential(0)}`;

  let warn: string | undefined;
  if (nClips > 3000) {
    warn = "Large dataset — voice cloning saturates around 1–2 hr; consider subsampling to your best ~1 500 clips for similar or better quality.";
  } else if (nClips < 50 && nClips > 0) {
    warn = "Small dataset — quality may be poor. Aim for at least 100 clips (~15 min processed audio).";
  }

  return { rank, epochs, lr, reason, warn };
}

export default function CreatorDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const creator = useSWR<Creator>(`/api/creators/${id}`, () =>
    fetch(`/api/creators/${id}`).then((r) => r.json()),
  );
  const jobs = useSWR<Job[]>(`/api/training/${id}/jobs`, () => api.listJobs(id), {
    refreshInterval: (data) =>
      data?.some((j) => j.status === "running" || j.status === "pending") ? 3000 : 30000,
  });
  const anyJobActive = jobs.data?.some(
    (j) => j.status === "running" || j.status === "pending",
  ) ?? false;

  const ppStatus = useSWR<{
    status: string; progress: number; log: string;
    n_input: number; n_output: number;
  }>(`/api/datasets/${id}/preprocess/status`, () => api.preprocessStatus(id), {
    refreshInterval: (d) => (d?.status === "running" ? 2000 : 30000),
  });
  const ppRunning = ppStatus.data?.status === "running";

  const raw = useSWR<{ name: string; size: number }[]>(
    `/api/datasets/${id}/raw`, () => api.listRaw(id),
    { refreshInterval: 60000 },
  );
  const clips = useSWR<Clip[]>(`/api/datasets/${id}/clips`, () => api.listClips(id), {
    refreshInterval: anyJobActive || ppRunning ? 5000 : 60000,
  });

  const sys = useSWR<SystemStatus>("/api/system/status", api.systemStatus, {
    refreshInterval: 0,
    revalidateOnFocus: false,
  });

  const totalRawSize = (raw.data ?? []).reduce((s, f) => s + f.size, 0);
  const totalClipSec = (clips.data ?? []).reduce((s, c) => s + c.duration, 0);

  const auto = autoLoraSettings(
    clips.data?.length ?? 0,
    totalClipSec,
    sys.data?.gpu.vram_gb ?? null,
    sys.data?.gpu.kind ?? "cpu",
  );

  const [training, setTraining] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<null | "preprocessing" | "waiting" | "starting">(null);
  const [pipelineErr, setPipelineErr] = useState<string | null>(null);
  const [useAuto, setUseAuto] = useState(true);
  const [epochs, setEpochs] = useState(8);
  const [rank, setRank] = useState(8);
  const [lr, setLr] = useState(1e-4);

  const effEpochs = useAuto ? auto.epochs : epochs;
  const effRank = useAuto ? auto.rank : rank;
  const effLr = useAuto ? auto.lr : lr;

  async function preprocess() {
    await api.preprocess(id);
    setTimeout(() => { clips.mutate(); raw.mutate(); }, 1500);
  }

  async function startTraining() {
    setTraining(true);
    setPipelineErr(null);
    try {
      await api.startTraining(id, {
        epochs: effEpochs, learning_rate: effLr, rank: effRank,
      });
      await jobs.mutate();
    } catch (e) {
      setPipelineErr((e as Error).message);
    } finally {
      setTraining(false);
    }
  }

  async function trainOnAll() {
    setTraining(true);
    setPipelineErr(null);
    try {
      let current = clips.data ?? [];

      if (current.length < 20 && (raw.data?.length ?? 0) > 0) {
        setPipelineStep("preprocessing");
        await api.preprocess(id);

        setPipelineStep("waiting");
        const start = Date.now();
        const TIMEOUT_MS = 30 * 60 * 1000;
        while (Date.now() - start < TIMEOUT_MS) {
          await new Promise((r) => setTimeout(r, 3000));
          const fresh = await api.listClips(id);
          await clips.mutate(fresh, { revalidate: false });
          if (fresh.length >= 20) {
            current = fresh;
            break;
          }
        }
        if (current.length < 20) {
          throw new Error(`Timed out waiting for preprocessing. Got ${current.length} clips after preprocess; need ≥20.`);
        }
      }

      if (current.length < 20) {
        throw new Error(`Need ≥20 clips to train. Have ${current.length}. Upload more audio.`);
      }

      const liveAuto = autoLoraSettings(
        current.length,
        current.reduce((s, c) => s + c.duration, 0),
        sys.data?.gpu.vram_gb ?? null,
        sys.data?.gpu.kind ?? "cpu",
      );

      setPipelineStep("starting");
      await api.startTraining(id, {
        epochs: useAuto ? liveAuto.epochs : epochs,
        learning_rate: useAuto ? liveAuto.lr : lr,
        rank: useAuto ? liveAuto.rank : rank,
      });
      await jobs.mutate();
    } catch (e) {
      setPipelineErr((e as Error).message);
    } finally {
      setTraining(false);
      setPipelineStep(null);
    }
  }

  async function signConsent() {
    await api.signConsent(id);
    creator.mutate();
  }

  const referenceCount = clips.data?.filter((c) => c.is_reference).length ?? 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">{creator.data?.name ?? id}</h1>
          <p className="text-sm text-[var(--color-muted)]">{id}</p>
        </div>
        <Link
          href={`/creators/${id}/synth`}
          className="bg-[var(--color-accent)] text-white px-4 py-2 rounded-md text-sm font-medium"
        >
          Synthesize →
        </Link>
      </div>

      {creator.data && !creator.data.consent_signed && (
        <Card className="border-amber-700/50">
          <CardTitle>Consent required</CardTitle>
          <p className="text-sm text-[var(--color-muted)] mb-3">
            Confirm the creator has signed the voice-cloning consent. Uploads and
            synthesis are blocked until this is checked.
          </p>
          <Button onClick={signConsent}>Mark consent signed</Button>
        </Card>
      )}

      <Card>
        <CardTitle>Upload audio</CardTitle>
        <UploadDropzone creatorId={id} onDone={() => { raw.mutate(); clips.mutate(); }} />
      </Card>

      <Card>
        <CardTitle>
          Uploaded files ({raw.data?.length ?? 0}
          {raw.data && raw.data.length > 0 ? ` · ${fmtBytes(totalRawSize)}` : ""})
        </CardTitle>
        {!raw.data || raw.data.length === 0 ? (
          <p className="text-sm text-[var(--color-muted)]">
            No raw files yet. Drop audio above.
          </p>
        ) : (
          <ul className="divide-y divide-[var(--color-border)] text-sm">
            {raw.data.map((f) => (
              <li key={f.name} className="flex items-center justify-between py-1.5">
                <span className="font-mono truncate flex-1 mr-3">{f.name}</span>
                <span className="text-[var(--color-muted)] w-24 text-right">{fmtBytes(f.size)}</span>
                <button
                  onClick={async () => {
                    if (!confirm(`Delete ${f.name}?`)) return;
                    await api.deleteRaw(id, f.name);
                    raw.mutate();
                  }}
                  className="ml-3 text-[var(--color-muted)] hover:text-red-400 px-2"
                  title="Delete file"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap items-center gap-3 mt-4 pt-4 border-t border-[var(--color-border)]">
          <Button
            variant="secondary"
            onClick={preprocess}
            disabled={!raw.data || raw.data.length === 0 || ppRunning}
          >
            {ppRunning ? "Preprocessing…" : "Run preprocessing"}
          </Button>
          <span className="text-xs text-[var(--color-muted)]">
            denoise → 24 kHz mono → VAD → loudness norm → ASR. Re-running replaces all clips for this creator.
          </span>
        </div>

        {ppStatus.data && ppStatus.data.status !== "idle" && (
          <div className="mt-4 pt-4 border-t border-[var(--color-border)]">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="font-semibold">Preprocess status</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${badge(ppStatus.data.status)}`}>
                {ppStatus.data.status}
              </span>
            </div>
            <div className="text-xs text-[var(--color-muted)] mb-2">
              files: {ppStatus.data.n_input} · clips produced: {ppStatus.data.n_output} ·
              progress: {Math.round(ppStatus.data.progress * 100)}%
            </div>
            <div className="h-1.5 bg-[var(--color-bg)] rounded mb-3">
              <div
                className="h-1.5 bg-[var(--color-accent)] rounded transition-all"
                style={{ width: `${Math.round(ppStatus.data.progress * 100)}%` }}
              />
            </div>
            <details className="text-xs">
              <summary className="cursor-pointer text-[var(--color-muted)] hover:text-white">
                Log ({(ppStatus.data.log.match(/\n/g)?.length ?? 0)} lines)
              </summary>
              <pre className="mt-2 text-[var(--color-muted)] max-h-64 overflow-auto whitespace-pre-wrap font-mono">
                {ppStatus.data.log || "—"}
              </pre>
            </details>
          </div>
        )}
      </Card>

      <Card>
        <CardTitle>
          Clips ({clips.data?.length ?? 0}) — references: {referenceCount}
        </CardTitle>
        {!clips.data || clips.data.length === 0 ? (
          <p className="text-sm text-[var(--color-muted)]">
            No clips yet. Upload + run preprocessing.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-[var(--color-muted)]">
                <tr>
                  <th className="py-2">Clip</th>
                  <th>Transcript (click to edit)</th>
                  <th>Sec</th>
                  <th>Emotion</th>
                  <th>Ref</th>
                  <th>Preview</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {clips.data.map((c) => (
                  <tr key={c.id} className="border-t border-[var(--color-border)]">
                    <td className="py-2 font-mono text-xs align-top">{c.id}</td>
                    <td className="align-top pr-3">
                      <textarea
                        defaultValue={c.text}
                        rows={2}
                        onBlur={(e) => {
                          if (e.target.value !== c.text) {
                            api.patchClip(c.id, { text: e.target.value });
                          }
                        }}
                        className="w-full min-w-[20rem] bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-xs"
                      />
                    </td>
                    <td className="align-top">{c.duration.toFixed(1)}</td>
                    <td className="align-top">
                      <select
                        defaultValue={c.emotion ?? ""}
                        onChange={(e) => api.patchClip(c.id, { emotion: e.target.value || null })}
                        className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-0.5"
                      >
                        <option value="">—</option>
                        {EMOTIONS.map((e) => <option key={e} value={e}>{e}</option>)}
                      </select>
                    </td>
                    <td className="align-top">
                      <input
                        type="checkbox" defaultChecked={c.is_reference}
                        onChange={(e) => api.patchClip(c.id, { is_reference: e.target.checked })}
                      />
                    </td>
                    <td className="align-top">
                      <audio controls src={`/files/${c.path}`} className="h-8" />
                    </td>
                    <td className="align-top">
                      <button
                        onClick={async () => {
                          if (!confirm(`Delete clip ${c.id}?`)) return;
                          await api.deleteClip(c.id);
                          clips.mutate();
                        }}
                        className="text-[var(--color-muted)] hover:text-red-400 px-2"
                        title="Delete clip"
                      >
                        ×
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold">LoRA training</h2>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox" checked={useAuto}
              onChange={(e) => setUseAuto(e.target.checked)}
            />
            <span className={useAuto ? "text-[var(--color-accent)]" : "text-[var(--color-muted)]"}>
              Auto settings
            </span>
          </label>
        </div>

        <p className="text-xs text-[var(--color-muted)] mb-2">{auto.reason}</p>
        {auto.warn && (
          <p className="text-xs text-amber-400 mb-4">{auto.warn}</p>
        )}

        <div className="flex flex-wrap items-end gap-4 mb-4">
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Epochs</div>
            <input
              type="number" value={effEpochs} min={1} max={50} disabled={useAuto}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 w-20 disabled:opacity-60"
            />
          </label>
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Rank</div>
            <input
              type="number" value={effRank} min={4} max={64} disabled={useAuto}
              onChange={(e) => setRank(parseInt(e.target.value))}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 w-20 disabled:opacity-60"
            />
          </label>
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Learning rate</div>
            <input
              type="number" value={effLr} min={1e-5} max={5e-4} step={1e-5} disabled={useAuto}
              onChange={(e) => setLr(parseFloat(e.target.value))}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 w-28 disabled:opacity-60"
            />
          </label>
          <Button
            onClick={trainOnAll}
            disabled={training || ((clips.data?.length ?? 0) < 20 && (raw.data?.length ?? 0) === 0)}
          >
            {pipelineStep === "preprocessing" ? "Preprocessing…" :
             pipelineStep === "waiting" ? "Waiting for clips…" :
             pipelineStep === "starting" ? "Starting training…" :
             training ? "Working…" :
             (clips.data?.length ?? 0) < 20 ? "Process + train all uploaded data" :
             "Train on all clips"}
          </Button>
          {(clips.data?.length ?? 0) >= 20 && (
            <Button variant="secondary" onClick={startTraining} disabled={training}>
              Train (skip preprocess)
            </Button>
          )}
        </div>

        {pipelineErr && (
          <p className="text-xs text-red-400 mb-3">Error: {pipelineErr}</p>
        )}
        {pipelineStep && (
          <p className="text-xs text-[var(--color-muted)] mb-3">
            Step: <span className="text-white">{pipelineStep}</span>
            {pipelineStep === "waiting" && ` · ${clips.data?.length ?? 0} clips so far`}
          </p>
        )}

        {jobs.data && jobs.data.length > 0 ? (
          <ul className="space-y-2">
            {jobs.data.map((j) => {
              const lines = j.log ? j.log.split("\n").filter(Boolean) : [];
              const ts = (s: string | null) => s ? new Date(s).toLocaleTimeString() : "—";
              return (
                <li key={j.id} className="border border-[var(--color-border)] rounded-md p-3">
                  <div className="flex justify-between items-start text-sm gap-3">
                    <div>
                      <div className="font-mono text-xs">{j.id}</div>
                      <div className="text-xs text-[var(--color-muted)] mt-0.5">
                        started {ts(j.started_at)}
                        {j.finished_at && ` · finished ${ts(j.finished_at)}`}
                      </div>
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap ${badge(j.status)}`}>
                      {j.status}
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 bg-[var(--color-bg)] rounded">
                    <div
                      className="h-1.5 bg-[var(--color-accent)] rounded transition-all"
                      style={{ width: `${Math.round(j.progress * 100)}%` }}
                    />
                  </div>
                  <div className="text-xs text-[var(--color-muted)] mt-1">
                    {Math.round(j.progress * 100)}%
                    {j.lora_path && ` · adapter: ${j.lora_path.split(/[\\/]/).slice(-2).join("/")}`}
                  </div>
                  {lines.length > 0 && (
                    <details className="mt-2" open={j.status === "running" || j.status === "failed"}>
                      <summary className="text-xs text-[var(--color-muted)] hover:text-white cursor-pointer">
                        Log ({lines.length} lines)
                      </summary>
                      <pre className="mt-2 text-xs text-[var(--color-muted)] max-h-80 overflow-auto whitespace-pre-wrap font-mono bg-[var(--color-bg)] rounded p-2">
                        {lines.slice(-200).join("\n")}
                      </pre>
                    </details>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <p className="text-sm text-[var(--color-muted)]">No training runs yet.</p>
        )}
      </Card>
    </div>
  );
}

function badge(status: string) {
  switch (status) {
    case "completed": return "bg-emerald-900/40 text-emerald-300";
    case "running":   return "bg-blue-900/40 text-blue-300";
    case "failed":    return "bg-red-900/40 text-red-300";
    default:          return "bg-zinc-800 text-zinc-300";
  }
}
