export function EvaluationBar({ whitePercent }: { whitePercent: number }) {
  const blackPercent = 100 - whitePercent;
  return (
    <div className="flex h-full min-h-[360px] w-8 overflow-hidden rounded-sm border border-zero-border bg-zinc-900">
      <div className="flex h-full w-full flex-col">
        <div className="bg-zinc-900 transition-[height]" style={{ height: `${blackPercent}%` }} />
        <div className="bg-zinc-100 transition-[height]" style={{ height: `${whitePercent}%` }} />
      </div>
    </div>
  );
}
