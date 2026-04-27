"use client";
import { use, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { AudioPlayer } from "@/components/audio-player";
import { api } from "@/lib/api";

export default function SynthPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const emotions = useSWR<string[]>("/api/tts/emotions", api.emotions);

  const [text, setText] = useState("");
  const [emotion, setEmotion] = useState("flirty");
  const [speed, setSpeed] = useState(1.0);
  const [seed, setSeed] = useState<string>("");
  const [useLora, setUseLora] = useState(true);
  const [busy, setBusy] = useState(false);
  const [outputs, setOutputs] = useState<{ url: string; emotion: string; seed: number | null }[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    setError(null);
    setBusy(true);
    try {
      const res = await api.synth({
        creator_id: id, text, emotion, speed,
        seed: seed ? parseInt(seed) : null, use_lora: useLora,
      });
      setOutputs((o) => [{ url: res.url, emotion: res.emotion, seed: res.seed }, ...o]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Synthesize</h1>
      <p className="text-sm text-[var(--color-muted)]">creator: <span className="font-mono">{id}</span></p>

      <Card>
        <CardTitle>Message</CardTitle>
        <textarea
          value={text} onChange={(e) => setText(e.target.value)}
          rows={4} maxLength={600}
          placeholder="Type the message text…"
          className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded-md px-3 py-2 text-sm"
        />

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Emotion</div>
            <select
              value={emotion} onChange={(e) => setEmotion(e.target.value)}
              className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1"
            >
              {(emotions.data ?? []).map((e) => <option key={e} value={e}>{e}</option>)}
            </select>
          </label>

          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Speed ({speed.toFixed(2)})</div>
            <input
              type="range" min={0.7} max={1.3} step={0.05}
              value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))}
              className="w-full"
            />
          </label>

          <label className="text-sm">
            <div className="text-[var(--color-muted)] mb-1">Seed (blank = random)</div>
            <input
              value={seed} onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              className="w-full bg-[var(--color-bg)] border border-[var(--color-border)] rounded px-2 py-1"
            />
          </label>

          <label className="text-sm flex items-end gap-2 pb-1">
            <input type="checkbox" checked={useLora} onChange={(e) => setUseLora(e.target.checked)} />
            <span>Use LoRA</span>
          </label>
        </div>

        <div className="mt-4 flex items-center gap-3">
          <Button onClick={generate} disabled={busy || !text.trim()}>
            {busy ? "Generating…" : "Generate"}
          </Button>
          {error && <span className="text-sm text-red-400">{error}</span>}
        </div>
      </Card>

      <Card>
        <CardTitle>Outputs</CardTitle>
        {outputs.length === 0 ? (
          <p className="text-sm text-[var(--color-muted)]">No generations yet.</p>
        ) : (
          <ul className="space-y-3">
            {outputs.map((o, i) => (
              <li key={i} className="border border-[var(--color-border)] rounded p-3">
                <div className="text-xs text-[var(--color-muted)] mb-1">
                  emotion={o.emotion} · seed={o.seed}
                </div>
                <AudioPlayer src={o.url} />
                <a href={o.url} download className="text-xs text-[var(--color-accent)] mt-1 inline-block">
                  Download
                </a>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
