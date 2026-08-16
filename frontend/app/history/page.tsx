"use client";

import { useEffect, useMemo, useState } from "react";
import type { Move } from "chess.js";
import { Chess } from "chess.js";
import { ChevronLeft, ChevronRight, RefreshCw } from "lucide-react";
import { Button } from "@/components/Button";
import { ZeroChessBoard, kingSquare } from "@/components/ChessBoard";

type TrainingGame = {
  id?: string;
  timestamp?: string;
  game_number?: number;
  game_index?: number;
  generation?: number;
  batch_index?: number;
  result: string;
  elo_after?: number;
  elo_delta?: number;
  rated_side?: string;
  replay_size?: number;
  train_step?: number;
  ply_count?: number;
  plies?: number;
  moves_san?: string[];
  moves?: string[];
  loss?: number;
  pgn?: string;
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
      const apiBase = process.env.NEXT_PUBLIC_ZERO_API_URL ?? "http://localhost:8765";
      const response = await fetch(`${apiBase}/history?limit=100`, { cache: "no-store" });
      if (!response.ok) throw new Error(`history request failed: ${response.status}`);
      const payload = (await response.json()) as { games: TrainingGame[] };
      const normalized = payload.games.map(normalizeGame);
      setGames(normalized);
      setSelectedId((current) => current ?? normalized[0]?.id ?? null);
    } catch {
      setGames([]);
      setSelectedId(null);
    } finally {
      setLoading(false);
    }
  }

  function selectGame(game: TrainingGame) {
    setSelectedId(game.id ?? null);
    setPly(game.ply_count ?? game.plies ?? gameMoves(game).length);
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
                   <span>Game {game.game_number ?? game.game_index ?? 0}</span>
                  <span>{game.result}</span>
                </div>
                <div className="mt-1 text-xs opacity-80">
                   Gen {game.generation ?? game.batch_index ?? 0} / Elo {(game.elo_after ?? 0).toFixed(1)} ({(game.elo_delta ?? 0) >= 0 ? "+" : ""}{(game.elo_delta ?? 0).toFixed(1)})
                </div>
                <div className="mt-1 truncate text-xs opacity-70">{gameMoves(game).slice(0, 10).join(" ")}</div>
              </button>
            ))}
            {!games.length && <div className="rounded-md bg-zero-panel2 px-3 py-4 text-sm text-zinc-400">No saved training games yet.</div>}
          </div>
        </aside>

        <section className="space-y-3">
          <div className="rounded-md bg-zero-panel px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                 <div className="text-sm font-semibold">{selected ? `Game ${selected.game_number ?? selected.game_index ?? 0}` : "No game selected"}</div>
                 <div className="text-xs text-zinc-400">{selected ? `${selected.ply_count ?? selected.plies ?? gameMoves(selected).length} plies / ${selected.result}` : "Start training to populate history"}</div>
              </div>
              <div className="text-right text-xs text-zinc-400">
                 <div>Replay {Math.min(ply, selected ? gameLength(selected) : 0)} / {selected ? gameLength(selected) : 0}</div>
                 <div>Loss {(selected?.loss ?? 0).toFixed(3)}</div>
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
             <Button icon={<ChevronRight size={17} />} disabled={!selected || ply >= (selected ? gameLength(selected) : 0)} onClick={() => setPly((value) => Math.min(selected ? gameLength(selected) : 0, value + 1))}>
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
  for (const notation of gameMoves(game).slice(0, Math.max(0, Math.min(ply, gameLength(game))))) {
    try {
      lastMove = chess.move(notation) as Move;
    } catch {
      if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/i.test(notation)) break;
      try {
        lastMove = chess.move({
          from: notation.slice(0, 2),
          to: notation.slice(2, 4),
          promotion: notation[4]?.toLowerCase() as "q" | "r" | "b" | "n" | undefined
        }) as Move;
      } catch {
        break;
      }
    }
  }
  return { game: chess, lastMove };
}

function gameMoves(game: TrainingGame): string[] {
  return game.moves_san ?? game.moves ?? [];
}

function gameLength(game: TrainingGame): number {
  return game.ply_count ?? game.plies ?? gameMoves(game).length;
}

function normalizeGame(game: TrainingGame, index: number): TrainingGame {
  return {
    ...game,
    id: game.id ?? `${game.batch_index ?? 0}-${game.game_index ?? index}`,
    game_number: game.game_number ?? game.game_index ?? index + 1,
    ply_count: game.ply_count ?? game.plies ?? gameMoves(game).length,
    moves_san: game.moves_san ?? game.moves ?? []
  };
}
