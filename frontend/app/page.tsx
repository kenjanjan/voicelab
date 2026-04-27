import Link from "next/link";
import { Card, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-semibold tracking-tight">
          Creator voice cloning, end to end.
        </h1>
        <p className="text-[var(--color-muted)] mt-2 max-w-2xl">
          Onboard a creator, upload audio, preprocess + transcribe, optionally fine-tune a
          LoRA on CosyVoice 2, and synthesize emotional voice messages. Local-first.
        </p>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardTitle>1. Onboard</CardTitle>
          <p className="text-sm text-[var(--color-muted)]">
            Create the creator and capture signed consent before any audio is uploaded.
          </p>
        </Card>
        <Card>
          <CardTitle>2. Dataset</CardTitle>
          <p className="text-sm text-[var(--color-muted)]">
            Upload raw audio, run denoise → VAD → ASR, then mark per-emotion reference
            clips.
          </p>
        </Card>
        <Card>
          <CardTitle>3. Train + Synthesize</CardTitle>
          <p className="text-sm text-[var(--color-muted)]">
            Run a LoRA fine-tune (optional) and generate voice messages with emotion
            control.
          </p>
        </Card>
      </section>

      <Link
        href="/creators"
        className="inline-block bg-[var(--color-accent)] text-white px-4 py-2 rounded-md font-medium"
      >
        Go to Creators →
      </Link>
    </div>
  );
}
