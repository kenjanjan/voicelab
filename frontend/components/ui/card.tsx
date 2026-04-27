import clsx from "clsx";
import { HTMLAttributes } from "react";

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      {...rest}
      className={clsx(
        "rounded-lg border border-[var(--color-border)] bg-[var(--color-panel)] p-5",
        className,
      )}
    />
  );
}

export function CardTitle({ className, ...rest }: HTMLAttributes<HTMLHeadingElement>) {
  return <h2 {...rest} className={clsx("text-base font-semibold mb-3", className)} />;
}
