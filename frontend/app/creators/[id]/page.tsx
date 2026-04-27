"use client";
import Link from "next/link";
import { use, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { UploadDropzone } from "@/components/upload-dropzone";
import { api, Clip, Creator, Job } from "@/lib/api";

const EMOTIONS = ["casual", "flirty", "seductive", "excited", "playful_giggle", "soft_intimate"];

export default function CreatorDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);

  const creator = useSWR<Creator>(`/api/creators/${id}`, () =>
    fetch(`/api/creators/${id}`).then((r) => r.json()),
  );
  const clips = useSWR<Clip[]>(`/api/datasets/${id}/clips`, () => api.listClips(id), {
    refreshInterval: 4000,
  });
  const jobs = useSWR<Job[]>(`/api/training/${id}/jobs`, () => api.listJobs(id), {
    refreshInterval: 3000,
  });

  const [training, setTraining] = useState(false);
  const [epochs, setEpochs] = useState(8);
  const [rank, setRank] = useState(16);

  async function preprocess() {
    await api.preprocess(id);
    setTimeout(() => clips.mutate(), 500);
  }

  async function startTraining() {
    setTraining(true);
    try {
      await api.startTraining(id, { epochs, learning_rate: 1e-4, rank });
      await jobs.mutate();
    } finally {
      setTraining(false);
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
        <UploadDropzone creatorId={id} onDone={() => clips.mutate()} />
        <div className="flex gap-3 mt-4">
          <Button variant="secondary" onClick={preprocess}>Run preprocessing</Button>
          <span className="text-xs text-[var(--color-muted)] self-center">
            denoise → 24k mono → VAD → loudness norm → ASR
          </span>
        </div>
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
                  <th>Transcript</th>
                  <th>Sec</th>
                  <th>Emotion</th>
                  <th>Reference</th>
                  <th>Preview</th>
                </tr>
              </thead>
              <tbody>
                {clips.data.map((c) => (
                  <tr key={c.id} className="border-t border-[var(--color-border)]">
                    <td className="py-2 font-mono text-xs">{c.id}</td>
                    <td className="max-w-[18rem] truncate" title={c.text}>{c.text || "—"}</td>
                    <td>{c.duration.toFixed(1)}</td>
                    <td>
                      <select
                        defaultValue={c.emotion ?? ""}
                        onChange={(e) => api.patchClip(c.id, { emotion: e.target.value || null })}
                        className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-0.5"
                      >
                        <option value="">—</option>
                        {EMOTIONS.map((e) => <option key={e} value={e}>{e}</option>)}
                      </select>
                    </td>
                    <td>
                      <input
                        type="checkbox" defaultChecked={c.is_reference}
                        onChange={(e) => api.patchClip(c.id, { is_reference: e.target.checked })}
                      />
                    </td>
                    <td>
                      <audio controls src={`/files/${c.path}`} className="h-8" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card>
        <CardTitle>LoRA training</CardTitle>
        <div className="flex flex-wrap items-end gap-4 mb-4">
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Epochs</div>
            <input type="number" value={epochs} min={1} max={50}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 w-20"
            />
          </label>
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Rank</div>
            <input type="number" value={rank} min={4} max={64}
              onChange={(e) => setRank(parseInt(e.target.value))}
              className="bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1 w-20"
            />
          </label>
          <Button onClick={startTraining} disabled={training}>
            {training ? "Starting…" : "Start training"}
          </Button>
        </div>

        {jobs.data && jobs.data.length > 0 ? (
          <ul className="space-y-2">
            {jobs.data.map((j) => (
              <li key={j.id} className="border border-[var(--color-border)] rounded-md p-3">
                <div className="flex justify-between text-sm">
                  <span className="font-mono">{j.id}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${badge(j.status)}`}>
                    {j.status}
                  </span>
                </div>
                <div className="mt-2 h-1.5 bg-[var(--color-bg)] rounded">
                  <div
                    className="h-1.5 bg-[var(--color-accent)] rounded"
                    style={{ width: `${Math.round(j.progress * 100)}%` }}
                  />
                </div>
                {j.log && (
                  <pre className="mt-2 text-xs text-[var(--color-muted)] max-h-32 overflow-auto whitespace-pre-wrap">
                    {j.log.split("\n").slice(-8).join("\n")}
                  </pre>
                )}
              </li>
            ))}
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
