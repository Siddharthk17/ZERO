import { formatClock } from "@/lib/chess";

export function Clock({ name, seconds, active }: { name: string; seconds: number; active?: boolean }) {
  return (
    <div className="flex items-center justify-between rounded-md bg-zero-panel px-4 py-3 shadow-sm">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-sm bg-[#5f5a54] text-sm font-bold text-white">
          {name.slice(0, 1)}
        </div>
        <div>
          <div className="text-sm font-semibold text-zinc-100">{name}</div>
          <div className="text-xs text-zinc-400">{active ? "to move" : "waiting"}</div>
        </div>
      </div>
      <div className={`rounded-md px-3 py-2 font-mono text-xl font-bold ${active ? "bg-zinc-100 text-zinc-950" : "bg-[#181715] text-zinc-300"}`}>
        {formatClock(seconds)}
      </div>
    </div>
  );
}
