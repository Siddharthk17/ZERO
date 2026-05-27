import Link from "next/link";
import { Eye, History, Play, Users } from "lucide-react";

const actions = [
  { href: "/play", label: "Play vs ZERO", icon: Play, active: true },
  { href: "/watch", label: "Watch ZERO vs ZERO", icon: Eye, active: true },
  { href: "/history", label: "Training History", icon: History, active: true },
  { href: "#", label: "Multiplayer", icon: Users, active: false }
];

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-zero-bg px-6">
      <div className="w-full max-w-md">
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-normal text-white">ZERO</h1>
          <div className="mt-2 text-sm text-zinc-400">Self-born chess intelligence</div>
        </div>
        <div className="space-y-3 rounded-md bg-zero-panel p-4 shadow-2xl">
          {actions.map(({ href, label, icon: Icon, active }) =>
            active ? (
              <Link
                key={label}
                href={href}
                className="flex w-full items-center gap-3 rounded-md bg-zero-panel2 px-5 py-4 text-left text-lg font-semibold text-zinc-100 hover:bg-[#3c3935]"
              >
                <Icon size={22} />
                {label}
              </Link>
            ) : (
              <button
                key={label}
                disabled
                className="flex w-full cursor-not-allowed items-center gap-3 rounded-md bg-[#34312d] px-5 py-4 text-left text-lg font-semibold text-zinc-500"
              >
                <Icon size={22} />
                {label}
                <span className="ml-auto text-xs uppercase tracking-wide">Coming soon</span>
              </button>
            )
          )}
        </div>
      </div>
    </main>
  );
}
