import type { Move } from "chess.js";
import { capturedPieces, getMaterialDifference } from "@/lib/chess";

const symbols: Record<"w" | "b", Record<string, string>> = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛" }
};

export function CapturedPieces({ history }: { history: Move[] }) {
  const captured = capturedPieces(history);
  const diff = getMaterialDifference(history);

  return (
    <section className="rounded-md bg-zero-panel px-4 py-3">
      <div className="mb-3 text-sm font-semibold text-zinc-200">Captured Material</div>
      <div className="space-y-3">
        {/* White pieces captured (held by Black) */}
        <div className="flex items-center justify-between min-h-7">
          <div className="flex items-center gap-1 text-2xl leading-none text-zinc-100">
            {captured.w.map((piece, idx) => (
              <span key={`w-${piece}-${idx}`} className="filter drop-shadow-sm select-none">
                {symbols.w[piece]}
              </span>
            ))}
          </div>
          {diff.blackAdvantage > 0 && (
            <span className="text-xs font-bold text-zinc-400 bg-zero-panel2 px-1.5 py-0.5 rounded">
              +{diff.blackAdvantage}
            </span>
          )}
        </div>

        {/* Black pieces captured (held by White) */}
        <div className="flex items-center justify-between min-h-7">
          <div className="flex items-center gap-1 text-2xl leading-none text-zinc-500">
            {captured.b.map((piece, idx) => (
              <span key={`b-${piece}-${idx}`} className="filter drop-shadow-sm select-none">
                {symbols.b[piece]}
              </span>
            ))}
          </div>
          {diff.whiteAdvantage > 0 && (
            <span className="text-xs font-bold text-zinc-400 bg-zero-panel2 px-1.5 py-0.5 rounded">
              +{diff.whiteAdvantage}
            </span>
          )}
        </div>
      </div>
    </section>
  );
}
