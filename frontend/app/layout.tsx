import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "VoiceLab",
  description: "Voice cloning + LoRA training for creator agencies",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-[var(--color-border)]">
          <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
            <Link href="/" className="font-semibold tracking-tight text-lg">
              <span className="text-[var(--color-accent)]">voice</span>lab
            </Link>
            <nav className="text-sm text-[var(--color-muted)] flex gap-6">
              <Link href="/creators" className="hover:text-white">Creators</Link>
              <a href="http://localhost:8000/docs" target="_blank" className="hover:text-white" rel="noreferrer">API</a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
