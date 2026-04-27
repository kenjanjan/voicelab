"use client";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Card, CardTitle } from "@/components/ui/card";
import { api, Creator } from "@/lib/api";

export default function CreatorsPage() {
  const { data, mutate, isLoading } = useSWR<Creator[]>("/api/creators", api.listCreators);
  const [name, setName] = useState("");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);

  async function create() {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api.createCreator(name.trim(), consent);
      setName("");
      setConsent(false);
      await mutate();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Creators</h1>

      <Card>
        <CardTitle>Onboard creator</CardTitle>
        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
          <input
            value={name} onChange={(e) => setName(e.target.value)}
            placeholder="Creator display name"
            className="flex-1 bg-[var(--color-bg)] border border-[var(--color-border)] rounded-md px-3 py-2 text-sm"
          />
          <label className="flex items-center gap-2 text-sm text-[var(--color-muted)]">
            <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            Consent signed
          </label>
          <Button onClick={create} disabled={busy || !name.trim()}>Create</Button>
        </div>
        <p className="text-xs text-[var(--color-muted)] mt-3">
          Consent is required before audio upload or synthesis. Track the signed
          agreement out-of-band.
        </p>
      </Card>

      <Card>
        <CardTitle>All creators</CardTitle>
        {isLoading ? (
          <p className="text-sm text-[var(--color-muted)]">Loading…</p>
        ) : !data || data.length === 0 ? (
          <p className="text-sm text-[var(--color-muted)]">No creators yet.</p>
        ) : (
          <ul className="divide-y divide-[var(--color-border)]">
            {data.map((c) => (
              <li key={c.id} className="py-2 flex items-center justify-between">
                <Link href={`/creators/${c.id}`} className="hover:text-[var(--color-accent)]">
                  <span className="font-medium">{c.name}</span>
                  <span className="text-xs text-[var(--color-muted)] ml-2">{c.id}</span>
                </Link>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full ${
                    c.consent_signed
                      ? "bg-emerald-900/40 text-emerald-300"
                      : "bg-amber-900/40 text-amber-300"
                  }`}
                >
                  {c.consent_signed ? "consent ✓" : "consent missing"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
