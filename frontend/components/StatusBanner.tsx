export function StatusBanner({ online }: { online: boolean }) {
  if (online) return null;
  return (
    <div className="fixed left-1/2 top-4 z-50 -translate-x-1/2 rounded-md border border-[#73523c] bg-[#3b2c22] px-4 py-2 text-sm font-semibold text-[#ffd9ad] shadow-xl">
      Engine offline
    </div>
  );
}
