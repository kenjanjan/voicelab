"use client";
import clsx from "clsx";
import { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "primary",
  className,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  const styles: Record<Variant, string> = {
    primary: "bg-[var(--color-accent)] text-white hover:opacity-90",
    secondary: "bg-[var(--color-panel)] border border-[var(--color-border)] hover:border-[var(--color-accent)]",
    ghost: "hover:bg-[var(--color-panel)] text-[var(--color-muted)] hover:text-white",
    danger: "bg-red-600/80 text-white hover:bg-red-600",
  };
  return (
    <button
      {...rest}
      className={clsx(
        "inline-flex items-center justify-center rounded-md px-3 py-1.5 text-sm font-medium",
        "transition disabled:opacity-50 disabled:cursor-not-allowed",
        styles[variant],
        className,
      )}
    />
  );
}
