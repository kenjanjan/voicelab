"use client";
import Link from "next/link";
import { use, useMemo, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { UploadDropzone } from "@/components/upload-dropzone";
import { api, Clip, Creator, Job, SystemStatus } from "@/lib/api";

const EMOTIONS = ["casual", "flirty", "seductive", "excited", "playful_giggle", "soft_intimate"];
const PAGE_SIZE = 50;

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function fmtDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(0)}s`;
  if (sec < 3600) return `${(sec / 60).toFixed(1)} min`;
  return `${(sec / 3600).toFixed(1)} hr`;
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

function badge(status: string) {
  switch (status) {
    case "completed": return "bg-emerald-900/40 text-emerald-300";
    case "running":   return "bg-blue-900/40 text-blue-300";
    case "pending":   return "bg-amber-900/40 text-amber-300";
    case "failed":    return "bg-red-900/40 text-red-300";
    default:          return "bg-zinc-800 text-zinc-300";
  }
}

type Tab = "dataset" | "training";

export default function CreatorDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [tab, setTab] = useState<Tab>("dataset");

  const creator = useSWR<Creator>(`/api/creators/${id}`, () =>
    fetch(`/api/creators/${id}`).then((r) => r.json()),
  );
  const jobs = useSWR<Job[]>(`/api/training/${id}/jobs`, () => api.listJobs(id), {
    refreshInterval: (data) =>
      data?.some((j) => j.status === "running" || j.status === "pending") ? 3000 : 30000,
  });
  const activeJob = jobs.data?.find((j) => j.status === "running" || j.status === "pending");

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
    refreshInterval: !!activeJob || ppRunning ? 5000 : 60000,
  });

  const sys = useSWR<SystemStatus>("/api/system/status", api.systemStatus, {
    refreshInterval: 0, revalidateOnFocus: false,
  });

  const totalRawSize = (raw.data ?? []).reduce((s, f) => s + f.size, 0);
  const totalClipSec = (clips.data ?? []).reduce((s, c) => s + c.duration, 0);
  const referenceCount = clips.data?.filter((c) => c.is_reference).length ?? 0;

  const auto = autoLoraSettings(
    clips.data?.length ?? 0, totalClipSec,
    sys.data?.gpu.vram_gb ?? null, sys.data?.gpu.kind ?? "cpu",
  );

  const [training, setTraining] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<null | "preprocessing" | "waiting" | "starting">(null);
  const [pipelineErr, setPipelineErr] = useState<string | null>(null);
  const [useAuto, setUseAuto] = useState(true);
  const [epochs, setEpochs] = useState(8);
  const [rank, setRank] = useState(8);
  const [lr, setLr] = useState(1e-4);
  const [device, setDevice] = useState<string>("");  // "" = auto

  const effEpochs = useAuto ? auto.epochs : epochs;
  const effRank = useAuto ? auto.rank : rank;
  const effLr = useAuto ? auto.lr : lr;

  async function preprocess() {
    await api.preprocess(id);
    setTimeout(() => { clips.mutate(); raw.mutate(); ppStatus.mutate(); }, 1000);
  }

  async function startTraining() {
    setTraining(true); setPipelineErr(null);
    try {
      await api.startTraining(id, {
        epochs: effEpochs, learning_rate: effLr, rank: effRank,
        device: device || null,
      });
      await jobs.mutate();
      setTab("training");
    } catch (e) {
      setPipelineErr((e as Error).message);
    } finally {
      setTraining(false);
    }
  }

  async function trainOnAll() {
    setTraining(true); setPipelineErr(null);
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
          if (fresh.length >= 20) { current = fresh; break; }
        }
        if (current.length < 20) {
          throw new Error(`Timed out waiting for preprocessing. Got ${current.length} clips; need ≥20.`);
        }
      }
      if (current.length < 20) {
        throw new Error(`Need ≥20 clips. Have ${current.length}. Upload more audio.`);
      }
      const liveAuto = autoLoraSettings(
        current.length,
        current.reduce((s, c) => s + c.duration, 0),
        sys.data?.gpu.vram_gb ?? null, sys.data?.gpu.kind ?? "cpu",
      );
      setPipelineStep("starting");
      await api.startTraining(id, {
        epochs: useAuto ? liveAuto.epochs : epochs,
        learning_rate: useAuto ? liveAuto.lr : lr,
        rank: useAuto ? liveAuto.rank : rank,
        device: device || null,
      });
      await jobs.mutate();
      setTab("training");
    } catch (e) {
      setPipelineErr((e as Error).message);
    } finally {
      setTraining(false); setPipelineStep(null);
    }
  }

  async function signConsent() {
    await api.signConsent(id);
    creator.mutate();
  }

  return (
    <div className="space-y-4">
      {/* Sticky header */}
      <div className="sticky top-0 z-20 -mx-6 px-6 py-3 bg-[var(--color-bg)]/95 backdrop-blur border-b border-[var(--color-border)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <Link href="/creators" className="text-xs text-[var(--color-muted)] hover:text-white">
              ← Creators
            </Link>
            <h1 className="text-xl font-semibold mt-0.5">{creator.data?.name ?? id}</h1>
            <p className="text-xs text-[var(--color-muted)] font-mono">{id}</p>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href={`/creators/${id}/synth`}
              className="bg-[var(--color-accent)] text-white px-3 py-1.5 rounded-md text-sm font-medium"
            >
              Synthesize →
            </Link>
          </div>
        </div>

        {/* Stats strip */}
        <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs">
          <Stat label="Raw" value={`${raw.data?.length ?? 0} · ${fmtBytes(totalRawSize)}`} />
          <Stat label="Clips" value={(clips.data?.length ?? 0).toLocaleString()} accent={(clips.data?.length ?? 0) >= 20} />
          <Stat label="References" value={referenceCount} accent={referenceCount > 0} />
          <Stat label="Audio" value={fmtDuration(totalClipSec)} />
          <Stat
            label="GPU"
            value={
              sys.data?.gpu.available
                ? `${sys.data.gpu.name}${sys.data.gpu.vram_gb ? ` · ${sys.data.gpu.vram_gb}GB` : ""}`
                : "CPU only"
            }
            accent={sys.data?.gpu.kind === "cuda"}
            warn={sys.data?.gpu.kind === "mps" || sys.data?.gpu.available === false}
          />
        </div>

        {/* Active job banner */}
        {(ppRunning || activeJob) && (
          <div className="mt-3 space-y-1">
            {ppRunning && ppStatus.data && (
              <ActivityBar
                label="Preprocessing"
                detail={`${ppStatus.data.n_output} clips so far · ${Math.round(ppStatus.data.progress * 100)}%`}
                progress={ppStatus.data.progress}
                onClick={() => setTab("dataset")}
              />
            )}
            {activeJob && (
              <ActivityBar
                label={`Training ${activeJob.id}`}
                detail={`${Math.round(activeJob.progress * 100)}% · ${activeJob.status}`}
                progress={activeJob.progress}
                onClick={() => setTab("training")}
              />
            )}
          </div>
        )}
      </div>

      {/* Consent banner */}
      {creator.data && !creator.data.consent_signed && (
        <Card className="border-amber-700/50">
          <CardTitle>Consent required</CardTitle>
          <p className="text-sm text-[var(--color-muted)] mb-3">
            Confirm the creator has signed the voice-cloning consent. Uploads and synthesis are blocked until this is checked.
          </p>
          <Button onClick={signConsent}>Mark consent signed</Button>
        </Card>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-[var(--color-border)]">
        <TabBtn active={tab === "dataset"} onClick={() => setTab("dataset")}>
          Dataset
        </TabBtn>
        <TabBtn active={tab === "training"} onClick={() => setTab("training")}>
          Training {jobs.data && jobs.data.length > 0 && (
            <span className="ml-1 text-xs text-[var(--color-muted)]">({jobs.data.length})</span>
          )}
        </TabBtn>
      </div>

      {tab === "dataset" && (
        <DatasetTab
          id={id}
          raw={raw}
          clips={clips}
          ppStatus={ppStatus}
          ppRunning={ppRunning}
          totalRawSize={totalRawSize}
          onPreprocess={preprocess}
          onResetPreprocess={async () => { await api.resetPreprocess(id); ppStatus.mutate(); }}
        />
      )}

      {tab === "training" && (
        <TrainingTab
          jobs={jobs}
          auto={auto}
          useAuto={useAuto} setUseAuto={setUseAuto}
          epochs={effEpochs} setEpochs={setEpochs}
          rank={effRank} setRank={setRank}
          lr={effLr} setLr={setLr}
          device={device} setDevice={setDevice}
          devices={sys.data?.gpu.devices ?? []}
          gpuKind={sys.data?.gpu.kind ?? "cpu"}
          onTrain={trainOnAll}
          onTrainQuick={startTraining}
          training={training}
          pipelineStep={pipelineStep}
          pipelineErr={pipelineErr}
          clipsCount={clips.data?.length ?? 0}
          rawCount={raw.data?.length ?? 0}
        />
      )}
    </div>
  );
}

/* ---------------- header bits ---------------- */

function Stat({ label, value, accent, warn }: {
  label: string; value: string | number; accent?: boolean; warn?: boolean;
}) {
  const dot = accent ? "bg-emerald-400" : warn ? "bg-amber-400" : "bg-zinc-500";
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${dot}`} />
      <span className="text-[var(--color-muted)]">{label}:</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function ActivityBar({ label, detail, progress, onClick }: {
  label: string; detail: string; progress: number; onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left rounded-md bg-[var(--color-panel)] border border-[var(--color-border)] hover:border-[var(--color-accent)] p-2 transition"
    >
      <div className="flex justify-between text-xs mb-1">
        <span className="font-medium">{label}</span>
        <span className="text-[var(--color-muted)]">{detail}</span>
      </div>
      <div className="h-1 bg-[var(--color-bg)] rounded">
        <div
          className="h-1 bg-[var(--color-accent)] rounded transition-all"
          style={{ width: `${Math.round(progress * 100)}%` }}
        />
      </div>
    </button>
  );
}

function TabBtn({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition ${
        active
          ? "border-[var(--color-accent)] text-white"
          : "border-transparent text-[var(--color-muted)] hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

/* ---------------- DATASET TAB ---------------- */

type SWRPair<T> = {
  data: T | undefined;
  mutate: () => void;
};

function DatasetTab({
  id, raw, clips, ppStatus, ppRunning, totalRawSize, onPreprocess, onResetPreprocess,
}: {
  id: string;
  raw: SWRPair<{ name: string; size: number }[]>;
  clips: SWRPair<Clip[]>;
  ppStatus: SWRPair<{ status: string; progress: number; log: string; n_input: number; n_output: number }>;
  ppRunning: boolean;
  totalRawSize: number;
  onPreprocess: () => void;
  onResetPreprocess: () => Promise<void>;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
      {/* Left column — upload + raw + preprocess */}
      <div className="lg:col-span-2 space-y-4">
        <Card>
          <CardTitle>Upload</CardTitle>
          <UploadDropzone creatorId={id} onDone={() => { raw.mutate(); clips.mutate(); }} />
        </Card>

        <Card>
          <CardTitle>
            Raw files {raw.data && raw.data.length > 0 && (
              <span className="font-normal text-xs text-[var(--color-muted)]">
                {raw.data.length} · {fmtBytes(totalRawSize)}
              </span>
            )}
          </CardTitle>
          {!raw.data || raw.data.length === 0 ? (
            <p className="text-sm text-[var(--color-muted)]">No raw files yet.</p>
          ) : (
            <div className="max-h-64 overflow-y-auto -mx-1 px-1">
              <ul className="divide-y divide-[var(--color-border)] text-sm">
                {raw.data.map((f) => (
                  <li key={f.name} className="flex items-center justify-between py-1.5 gap-2">
                    <span className="font-mono truncate flex-1 text-xs" title={f.name}>{f.name}</span>
                    <span className="text-[var(--color-muted)] text-xs whitespace-nowrap">{fmtBytes(f.size)}</span>
                    <button
                      onClick={async () => {
                        if (!confirm(`Delete ${f.name}?`)) return;
                        await api.deleteRaw(id, f.name);
                        raw.mutate();
                      }}
                      className="text-[var(--color-muted)] hover:text-red-400 px-1"
                      title="Delete"
                    >×</button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
            <Button
              variant="secondary"
              onClick={onPreprocess}
              disabled={!raw.data || raw.data.length === 0 || ppRunning}
            >
              {ppRunning ? "Preprocessing…" : "Run preprocessing"}
            </Button>
            <p className="text-xs text-[var(--color-muted)] mt-2">
              denoise → 24kHz mono → VAD → loudness norm → ASR
            </p>
          </div>
        </Card>

        {ppStatus.data && ppStatus.data.status !== "idle" && (
          <Card>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold">Preprocess status</h3>
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded-full ${badge(ppStatus.data.status)}`}>
                  {ppStatus.data.status}
                </span>
                {ppStatus.data.status === "running" && (
                  <button
                    onClick={async () => {
                      if (confirm("Mark this preprocess job as failed? Use this only if it's actually stuck (e.g. backend was killed).")) {
                        await onResetPreprocess();
                      }
                    }}
                    className="text-xs text-amber-400 hover:text-amber-300"
                    title="Force mark as failed"
                  >reset</button>
                )}
              </div>
            </div>
            <div className="text-xs text-[var(--color-muted)] mb-2">
              files: {ppStatus.data.n_input} · clips: {ppStatus.data.n_output} · {Math.round(ppStatus.data.progress * 100)}%
            </div>
            <div className="h-1.5 bg-[var(--color-bg)] rounded mb-3">
              <div
                className="h-1.5 bg-[var(--color-accent)] rounded transition-all"
                style={{ width: `${Math.round(ppStatus.data.progress * 100)}%` }}
              />
            </div>
            <details open={ppStatus.data.status === "running" || ppStatus.data.status === "failed"}>
              <summary className="text-xs text-[var(--color-muted)] hover:text-white cursor-pointer">
                Log ({(ppStatus.data.log.match(/\n/g)?.length ?? 0)} lines)
              </summary>
              <pre className="mt-2 text-[10px] text-[var(--color-muted)] max-h-80 overflow-auto whitespace-pre-wrap font-mono bg-[var(--color-bg)] rounded p-2">
                {ppStatus.data.log || "—"}
              </pre>
            </details>
          </Card>
        )}
      </div>

      {/* Right column — clips browser */}
      <div className="lg:col-span-3">
        <ClipsBrowser id={id} clips={clips} />
      </div>
    </div>
  );
}

/* ---------------- CLIPS BROWSER ---------------- */

function ClipsBrowser({ id, clips }: {
  id: string;
  clips: SWRPair<Clip[]>;
}) {
  const [search, setSearch] = useState("");
  const [emotionFilter, setEmotionFilter] = useState("");
  const [refOnly, setRefOnly] = useState(false);
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    let list = clips.data ?? [];
    if (search) {
      const q = search.toLowerCase();
      list = list.filter((c) => c.text.toLowerCase().includes(q) || String(c.id).includes(q));
    }
    if (emotionFilter) list = list.filter((c) => c.emotion === emotionFilter);
    if (refOnly) list = list.filter((c) => c.is_reference);
    return list;
  }, [clips.data, search, emotionFilter, refOnly]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageClips = filtered.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  const refCount = (clips.data ?? []).filter((c) => c.is_reference).length;

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-semibold">
          Clips {filtered.length !== (clips.data?.length ?? 0)
            ? <span className="text-[var(--color-muted)] text-sm font-normal">({filtered.length} of {clips.data?.length ?? 0})</span>
            : <span className="text-[var(--color-muted)] text-sm font-normal">({clips.data?.length ?? 0})</span>}
        </h2>
        <span className="text-xs text-[var(--color-muted)]">
          {refCount} marked as reference
        </span>
      </div>

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2 mb-3 pb-3 border-b border-[var(--color-border)]">
        <input
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          placeholder="Search transcript…"
          className="flex-1 min-w-[12rem] bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm"
        />
        <select
          value={emotionFilter}
          onChange={(e) => { setEmotionFilter(e.target.value); setPage(0); }}
          className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm"
        >
          <option value="">All emotions</option>
          <option value="__none__" disabled>—</option>
          {EMOTIONS.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <label className="text-xs flex items-center gap-1.5">
          <input
            type="checkbox" checked={refOnly}
            onChange={(e) => { setRefOnly(e.target.checked); setPage(0); }}
          />
          References only
        </label>
        {(search || emotionFilter || refOnly) && (
          <button
            onClick={() => { setSearch(""); setEmotionFilter(""); setRefOnly(false); setPage(0); }}
            className="text-xs text-[var(--color-muted)] hover:text-white"
          >Clear</button>
        )}
      </div>

      {!clips.data || clips.data.length === 0 ? (
        <p className="text-sm text-[var(--color-muted)] py-8 text-center">
          No clips yet. Upload audio and run preprocessing.
        </p>
      ) : pageClips.length === 0 ? (
        <p className="text-sm text-[var(--color-muted)] py-8 text-center">
          No matches.
        </p>
      ) : (
        <>
          <ul className="divide-y divide-[var(--color-border)]">
            {pageClips.map((c) => (
              <ClipRow key={c.id} clip={c} onChange={() => clips.mutate()} />
            ))}
          </ul>

          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-[var(--color-border)] text-xs">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={safePage === 0}
                className="px-3 py-1 rounded border border-[var(--color-border)] disabled:opacity-40"
              >← Prev</button>
              <span className="text-[var(--color-muted)]">
                Page {safePage + 1} of {totalPages} · showing {safePage * PAGE_SIZE + 1}–{Math.min((safePage + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={safePage >= totalPages - 1}
                className="px-3 py-1 rounded border border-[var(--color-border)] disabled:opacity-40"
              >Next →</button>
            </div>
          )}
        </>
      )}
    </Card>
  );
}

function ClipRow({ clip, onChange }: { clip: Clip; onChange: () => void }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <li className="py-2">
      <div className="flex items-center gap-2 text-sm">
        <span className="font-mono text-xs text-[var(--color-muted)] w-12 shrink-0">#{clip.id}</span>
        <span
          className="flex-1 truncate text-xs cursor-pointer"
          onClick={() => setExpanded((e) => !e)}
          title="Click to expand"
        >
          {clip.text || <span className="text-[var(--color-muted)] italic">no transcript</span>}
        </span>
        <span className="text-[10px] text-[var(--color-muted)] w-10 text-right">{clip.duration.toFixed(1)}s</span>
        <select
          value={clip.emotion ?? ""}
          onChange={async (e) => { await api.patchClip(clip.id, { emotion: e.target.value || null }); onChange(); }}
          className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-1.5 py-0.5 text-xs"
        >
          <option value="">—</option>
          {EMOTIONS.map((e) => <option key={e} value={e}>{e}</option>)}
        </select>
        <label className="flex items-center gap-1 text-xs" title="Use as reference for synthesis">
          <input
            type="checkbox" defaultChecked={clip.is_reference}
            onChange={async (e) => { await api.patchClip(clip.id, { is_reference: e.target.checked }); onChange(); }}
          />
          ref
        </label>
        <button
          onClick={() => setExpanded((e) => !e)}
          className="text-[var(--color-muted)] hover:text-white text-xs px-1"
        >
          {expanded ? "▾" : "▸"}
        </button>
        <button
          onClick={async () => {
            if (!confirm(`Delete clip ${clip.id}?`)) return;
            await api.deleteClip(clip.id);
            onChange();
          }}
          className="text-[var(--color-muted)] hover:text-red-400 px-1"
          title="Delete"
        >×</button>
      </div>

      {expanded && (
        <div className="mt-2 ml-14 space-y-2">
          <textarea
            defaultValue={clip.text}
            rows={2}
            onBlur={async (e) => {
              if (e.target.value !== clip.text) {
                await api.patchClip(clip.id, { text: e.target.value });
                onChange();
              }
            }}
            className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-xs"
            placeholder="Transcript"
          />
          <audio controls src={`/files/${clip.path}`} className="w-full h-8" />
        </div>
      )}
    </li>
  );
}

/* ---------------- TRAINING TAB ---------------- */

function TrainingTab({
  jobs, auto, useAuto, setUseAuto,
  epochs, setEpochs, rank, setRank, lr, setLr,
  device, setDevice, devices, gpuKind,
  onTrain, onTrainQuick, training, pipelineStep, pipelineErr,
  clipsCount, rawCount,
}: {
  jobs: SWRPair<Job[]>;
  auto: ReturnType<typeof autoLoraSettings>;
  useAuto: boolean; setUseAuto: (v: boolean) => void;
  epochs: number; setEpochs: (v: number) => void;
  rank: number; setRank: (v: number) => void;
  lr: number; setLr: (v: number) => void;
  device: string; setDevice: (v: string) => void;
  devices: { id: string; name: string; vram_gb: number | null }[];
  gpuKind: string;
  onTrain: () => void;
  onTrainQuick: () => void;
  training: boolean;
  pipelineStep: null | "preprocessing" | "waiting" | "starting";
  pipelineErr: string | null;
  clipsCount: number; rawCount: number;
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
      {/* Settings + start */}
      <div className="lg:col-span-2 space-y-4">
        <Card>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-semibold">LoRA settings</h2>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox" checked={useAuto}
                onChange={(e) => setUseAuto(e.target.checked)}
              />
              <span className={useAuto ? "text-[var(--color-accent)]" : "text-[var(--color-muted)]"}>
                Auto
              </span>
            </label>
          </div>

          <p className="text-xs text-[var(--color-muted)] mb-2">{auto.reason}</p>
          {auto.warn && <p className="text-xs text-amber-400 mb-3">{auto.warn}</p>}

          <div className="grid grid-cols-3 gap-3 mb-3">
            <label className="text-xs">
              <div className="text-[var(--color-muted)] mb-1">Epochs</div>
              <input
                type="number" value={epochs} min={1} max={50} disabled={useAuto}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm disabled:opacity-60"
              />
            </label>
            <label className="text-xs">
              <div className="text-[var(--color-muted)] mb-1">Rank</div>
              <input
                type="number" value={rank} min={4} max={64} disabled={useAuto}
                onChange={(e) => setRank(parseInt(e.target.value))}
                className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm disabled:opacity-60"
              />
            </label>
            <label className="text-xs">
              <div className="text-[var(--color-muted)] mb-1">LR</div>
              <input
                type="number" value={lr} min={1e-5} max={5e-4} step={1e-5} disabled={useAuto}
                onChange={(e) => setLr(parseFloat(e.target.value))}
                className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm disabled:opacity-60"
              />
            </label>
          </div>

          <label className="text-xs block mb-4">
            <div className="text-[var(--color-muted)] mb-1">Device</div>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 text-sm"
            >
              <option value="">Auto ({gpuKind === "cuda" ? "GPU" : gpuKind === "mps" ? "MPS" : "CPU"})</option>
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id} — {d.name}{d.vram_gb ? ` (${d.vram_gb} GB)` : ""}
                </option>
              ))}
            </select>
            {gpuKind === "cpu" && (
              <p className="text-amber-400 mt-1">
                No GPU detected. Install CUDA torch in the venv: <span className="font-mono">pip uninstall -y torch torchaudio &amp;&amp; pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124</span>, then restart uvicorn.
              </p>
            )}
          </label>

          <div className="flex flex-col gap-2">
            <Button
              onClick={onTrain}
              disabled={training || (clipsCount < 20 && rawCount === 0)}
            >
              {pipelineStep === "preprocessing" ? "Preprocessing…" :
               pipelineStep === "waiting" ? "Waiting for clips…" :
               pipelineStep === "starting" ? "Starting…" :
               training ? "Working…" :
               clipsCount < 20 ? "Process + train all data" :
               "Train on all clips"}
            </Button>
            {clipsCount >= 20 && (
              <Button variant="secondary" onClick={onTrainQuick} disabled={training}>
                Train (skip preprocess)
              </Button>
            )}
          </div>

          {pipelineErr && (
            <p className="text-xs text-red-400 mt-3">{pipelineErr}</p>
          )}
          {pipelineStep && (
            <p className="text-xs text-[var(--color-muted)] mt-3">
              Pipeline: <span className="text-white">{pipelineStep}</span>
            </p>
          )}
        </Card>
      </div>

      {/* Job history */}
      <div className="lg:col-span-3">
        <Card>
          <CardTitle>
            Training jobs {jobs.data && jobs.data.length > 0 && (
              <span className="text-xs font-normal text-[var(--color-muted)]">({jobs.data.length})</span>
            )}
          </CardTitle>
          {!jobs.data || jobs.data.length === 0 ? (
            <p className="text-sm text-[var(--color-muted)] py-8 text-center">
              No training runs yet.
            </p>
          ) : (
            <ul className="space-y-3">
              {jobs.data.map((j) => (
                <JobCard
                  key={j.id} job={j}
                  onReset={async () => {
                    if (!confirm(`Reset job ${j.id} to failed? Only do this if the actual process is dead (e.g. backend was killed).`)) return;
                    await api.resetJob(j.id);
                    jobs.mutate();
                  }}
                />
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  );
}

function JobCard({ job, onReset }: { job: Job; onReset: () => Promise<void> }) {
  const lines = job.log ? job.log.split("\n").filter(Boolean) : [];
  const ts = (s: string | null) => s ? new Date(s).toLocaleString() : "—";
  const stuck = job.status === "running" || job.status === "pending";
  return (
    <li className="border border-[var(--color-border)] rounded-md p-3">
      <div className="flex justify-between items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-xs">{job.id}</div>
          <div className="text-xs text-[var(--color-muted)] mt-0.5">
            started {ts(job.started_at)}
            {job.finished_at && ` · finished ${ts(job.finished_at)}`}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-xs px-2 py-0.5 rounded-full whitespace-nowrap ${badge(job.status)}`}>
            {job.status}
          </span>
          {stuck && (
            <button
              onClick={onReset}
              className="text-xs text-amber-400 hover:text-amber-300"
              title="Force mark as failed (use only if process is actually dead)"
            >reset</button>
          )}
        </div>
      </div>

      <div className="mt-2 h-1.5 bg-[var(--color-bg)] rounded">
        <div
          className="h-1.5 bg-[var(--color-accent)] rounded transition-all"
          style={{ width: `${Math.round(job.progress * 100)}%` }}
        />
      </div>
      <div className="text-xs text-[var(--color-muted)] mt-1">
        {Math.round(job.progress * 100)}%
        {job.lora_path && ` · adapter: ${job.lora_path.split(/[\\/]/).slice(-2).join("/")}`}
      </div>

      {lines.length > 0 && (
        <details className="mt-2" open={job.status === "running" || job.status === "failed"}>
          <summary className="text-xs text-[var(--color-muted)] hover:text-white cursor-pointer">
            Log · {lines.length} lines
          </summary>
          <pre className="mt-2 text-[10px] text-[var(--color-muted)] max-h-96 overflow-auto whitespace-pre-wrap font-mono bg-[var(--color-bg)] rounded p-2">
            {lines.slice(-300).join("\n")}
          </pre>
        </details>
      )}
    </li>
  );
}
