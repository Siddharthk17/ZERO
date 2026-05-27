import type { Move } from "chess.js";
import { capturedPieces } from "@/lib/chess";

const symbols: Record<string, string> = {
  p: "♟",
  n: "♞",
  b: "♝",
  r: "♜",
  q: "♛"
};

export function CapturedPieces({ history }: { history: Move[] }) {
  const captured = capturedPieces(history);
  return (
    <section className="rounded-md bg-zero-panel px-4 py-3">
      <div className="mb-3 text-sm font-semibold text-zinc-200">Captured</div>
      <div className="space-y-2 text-2xl leading-none">
        <div className="min-h-7 text-zinc-100">{captured.w.map((piece, idx) => <span key={`${piece}-${idx}`}>{symbols[piece]}</span>)}</div>
        <div className="min-h-7 text-zinc-500">{captured.b.map((piece, idx) => <span key={`${piece}-${idx}`}>{symbols[piece]}</span>)}</div>
      </div>
    </section>
  );
}
