import type { Move } from "chess.js";

export function MoveHistory({ history }: { history: Move[] }) {
  const rows = [];
  for (let i = 0; i < history.length; i += 2) {
    rows.push({ move: i / 2 + 1, white: history[i]?.san, black: history[i + 1]?.san });
  }
  return (
    <section className="min-h-0 rounded-md bg-zero-panel">
      <div className="border-b border-zero-border px-4 py-3 text-sm font-semibold text-zinc-200">Moves</div>
      <div className="scrollbar-dark max-h-72 overflow-y-auto text-sm md:max-h-[420px]">
        {rows.length === 0 ? (
          <div className="px-4 py-5 text-zinc-500">No moves yet</div>
        ) : (
          rows.map((row) => (
            <div key={row.move} className="grid grid-cols-[48px_1fr_1fr] border-b border-[#34312d] text-zinc-200">
              <div className="bg-[#211f1c] px-3 py-2 text-zinc-500">{row.move}.</div>
              <div className="px-3 py-2">{row.white}</div>
              <div className="px-3 py-2">{row.black}</div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
