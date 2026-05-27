"use client";

import { useEffect, useMemo, useState } from "react";
import type { Move } from "chess.js";
import { Chess } from "chess.js";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { Button } from "@/components/Button";
import { ZeroChessBoard, kingSquare } from "@/components/ChessBoard";

type TrainingGame = {
  id: string;
  timestamp: string;
  game_number: number;
  generation: number;
  result: string;
  elo_after: number;
  elo_delta: number;
  rated_side: string;
  replay_size: number;
  train_step: number;
  ply_count: number;
  moves_san: string[];
  loss: number;
  pgn: string;
};

export default function HistoryPage() {
  const [games, setGames] = useState<TrainingGame[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [ply, setPly] = useState(0);
  const [loading, setLoading] = useState(true);
  const selected = games.find((game) => game.id === selectedId) ?? games[0] ?? null;
  const replay = useMemo(() => buildReplay(selected, ply), [selected, ply]);

  useEffect(() => {
    loadHistory();
    const timer = window.setInterval(loadHistory, 10000);
    return () => window.clearInterval(timer);
  }, []);

  async function loadHistory() {
    try {
      const response = await fetch("http://localhost:8765/history?limit=100", { cache: "no-store" });
      const payload = (await response.json()) as { games: TrainingGame[] };
      setGames(payload.games);
      setSelectedId((current) => current ?? payload.games[0]?.id ?? null);
    } finally {
      setLoading(false);
    }
  }

  function selectGame(game: TrainingGame) {
    setSelectedId(game.id);
    setPly(game.ply_count);
  }

  return (
    <main className="min-h-screen bg-zero-bg px-3 py-4 text-zinc-100 md:px-8">
      <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-[360px_minmax(340px,720px)_360px]">
        <aside className="rounded-md bg-zero-panel p-3">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold">Training games</div>
              <div className="text-xs text-zinc-400">{loading ? "loading" : `${games.length} saved`}</div>
            </div>
            <Button icon={<RefreshCw size={16} />} onClick={loadHistory}>
              Refresh
            </Button>
          </div>
          <div className="max-h-[72vh] space-y-2 overflow-y-auto pr-1">
            {games.map((game) => (
              <button
                key={game.id}
                onClick={() => selectGame(game)}
                className={`w-full rounded-md px-3 py-3 text-left ${selected?.id === game.id ? "bg-zero-accent text-white" : "bg-zero-panel2 text-zinc-200 hover:bg-[#3c3935]"}`}
              >
                <div className="flex items-center justify-between text-sm font-semibold">
                  <span>Game {game.game_number}</span>
                  <span>{game.result}</span>
                </div>
                <div className="mt-1 text-xs opacity-80">
                  Gen {game.generation} / Elo {game.elo_after.toFixed(1)} ({game.elo_delta >= 0 ? "+" : ""}{game.elo_delta.toFixed(1)})
                </div>
                <div className="mt-1 truncate text-xs opacity-70">{game.moves_san.slice(0, 10).join(" ")}</div>
              </button>
            ))}
            {!games.length && <div className="rounded-md bg-zero-panel2 px-3 py-4 text-sm text-zinc-400">No saved training games yet.</div>}
          </div>
        </aside>

        <section className="space-y-3">
          <div className="rounded-md bg-zero-panel px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold">{selected ? `Game ${selected.game_number}` : "No game selected"}</div>
                <div className="text-xs text-zinc-400">{selected ? `${selected.ply_count} plies / ${selected.result}` : "Start training to populate history"}</div>
              </div>
              <div className="text-right text-xs text-zinc-400">
                <div>Replay {Math.min(ply, selected?.ply_count ?? 0)} / {selected?.ply_count ?? 0}</div>
                <div>Loss {selected?.loss.toFixed(3) ?? "0.000"}</div>
              </div>
            </div>
          </div>
          <ZeroChessBoard
            game={replay.game}
            fen={replay.game.fen()}
            orientation="white"
            selectedSquare={null}
            legalMoves={[]}
            lastMove={replay.lastMove}
            checkSquare={kingSquare(replay.game)}
            flashSquare={null}
            disabled
            onPieceDrop={() => false}
            onSquareClick={() => undefined}
          />
          <div className="grid grid-cols-3 gap-2 rounded-md bg-zero-panel p-3">
            <Button icon={<ChevronLeft size={17} />} disabled={!selected || ply <= 0} onClick={() => setPly((value) => Math.max(0, value - 1))}>
              Back
            </Button>
            <Button disabled={!selected} onClick={() => setPly(0)}>
              Start
            </Button>
            <Button icon={<ChevronRight size={17} />} disabled={!selected || ply >= selected.ply_count} onClick={() => setPly((value) => Math.min(selected?.ply_count ?? 0, value + 1))}>
              Next
            </Button>
          </div>
        </section>

        <aside className="rounded-md bg-zero-panel p-4">
          <div className="mb-3 text-sm font-semibold">PGN</div>
          <pre className="max-h-[78vh] overflow-y-auto whitespace-pre-wrap text-xs leading-5 text-zinc-300">{selected?.pgn ?? "No PGN saved yet."}</pre>
        </aside>
      </div>
    </main>
  );
}

function buildReplay(game: TrainingGame | null, ply: number) {
  const chess = new Chess();
  let lastMove: Move | null = null;
  if (!game) return { game: chess, lastMove };
  for (const san of game.moves_san.slice(0, Math.max(0, Math.min(ply, game.ply_count)))) {
    try {
      lastMove = chess.move(san) as Move;
    } catch {
      break;
    }
  }
  return { game: chess, lastMove };
}
