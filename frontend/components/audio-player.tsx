"use client";

export function AudioPlayer({ src }: { src: string }) {
  return (
    <audio
      controls
      src={src}
      className="w-full mt-2 rounded"
      style={{ filter: "invert(0.92)" }}
    />
  );
}
