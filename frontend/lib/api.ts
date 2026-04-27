export type Creator = { id: string; name: string; consent_signed: boolean };
export type Clip = {
  id: number; path: string; text: string;
  emotion: string | null; duration: number; is_reference: boolean;
};
export type Job = {
  id: string; creator_id: string; status: string; progress: number;
  log: string; started_at: string | null; finished_at: string | null;
  lora_path: string | null;
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export type SystemStatus = {
  backend_ok: boolean;
  gpu: { available: boolean; kind: string; name: string; vram_gb: number | null };
  models: {
    cosyvoice_repo: boolean; cosyvoice_repo_path: string;
    cosyvoice_weights: boolean; cosyvoice_weights_path: string;
  };
  storage: Record<string, string>;
  counts: { creators: number; clips: number; reference_clips: number };
  setup: { status: string; step: string | null; progress: number; log: string };
};

export const api = {
  systemStatus: () => req<SystemStatus>("/api/system/status"),
  triggerSetup: () => req<{ status: string }>("/api/system/setup", { method: "POST" }),

  listCreators: () => req<Creator[]>("/api/creators"),
  createCreator: (name: string, consent_signed: boolean) =>
    req<Creator>("/api/creators", { method: "POST", body: JSON.stringify({ name, consent_signed }) }),
  signConsent: (id: string) =>
    req<Creator>(`/api/creators/${id}/consent`, { method: "PATCH" }),

  listClips: (id: string) => req<Clip[]>(`/api/datasets/${id}/clips`),
  preprocess: (id: string) =>
    req<{ status: string }>(`/api/datasets/${id}/preprocess`, { method: "POST" }),
  patchClip: (clipId: number, body: Partial<Pick<Clip, "emotion" | "is_reference" | "text">>) =>
    req<Clip>(`/api/datasets/clips/${clipId}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteClip: (clipId: number) =>
    fetch(`/api/datasets/clips/${clipId}`, { method: "DELETE" }).then((r) => {
      if (!r.ok && r.status !== 204) throw new Error(`${r.status}`);
    }),

  startTraining: (id: string, body: { epochs: number; learning_rate: number; rank: number }) =>
    req<Job>(`/api/training/${id}/start`, { method: "POST", body: JSON.stringify(body) }),
  listJobs: (id: string) => req<Job[]>(`/api/training/${id}/jobs`),
  getJob: (jobId: string) => req<Job>(`/api/training/jobs/${jobId}`),

  emotions: () => req<string[]>("/api/tts/emotions"),
  synth: (body: {
    creator_id: string; text: string; emotion: string;
    speed?: number; seed?: number | null; use_lora?: boolean;
  }) => req<{ url: string; filename: string; emotion: string; seed: number | null }>(
    "/api/tts", { method: "POST", body: JSON.stringify(body) },
  ),
};

export async function uploadAudio(creatorId: string, files: File[]) {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const res = await fetch(`/api/datasets/${creatorId}/upload`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<{ saved: string[]; count: number }>;
}
