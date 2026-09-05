import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroSplit — Predicted Cortical Response",
  description:
    "In-silico neural content experimentation. Visualizes cortical fMRI responses predicted by Meta's TRIBE v2 — model predictions, not measurements from any viewer.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="app-shell min-h-screen antialiased">{children}</body>
    </html>
  );
}
