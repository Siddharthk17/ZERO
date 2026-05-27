import { RotateCcw } from "lucide-react";
import { Button } from "./Button";

export function GameEndModal({ result, onRematch }: { result: string | null; onRematch: () => void }) {
  if (!result) return null;
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/70 px-4">
      <div className="w-full max-w-sm rounded-md border border-zero-border bg-zero-panel p-6 shadow-2xl">
        <div className="mb-2 text-2xl font-bold text-white">{result}</div>
        <div className="mb-6 text-sm text-zinc-400">Game complete</div>
        <Button variant="primary" icon={<RotateCcw size={18} />} onClick={onRematch} className="w-full">
          Rematch
        </Button>
      </div>
    </div>
  );
}
