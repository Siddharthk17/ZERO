import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ZERO Chess",
  description: "Play and watch the ZERO self-play chess engine."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
