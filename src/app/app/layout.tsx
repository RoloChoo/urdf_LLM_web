// src/app/layout.tsx
import "./globals.css";
import type { Metadata } from "next";
import AiVersionHud from "@/components/AiVersionHud";

export const metadata: Metadata = {
  title: "Mechaverse",
  description: "Mechaverse URDF Viewer",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        {children}
        <AiVersionHud />
      </body>
    </html>
  );
}