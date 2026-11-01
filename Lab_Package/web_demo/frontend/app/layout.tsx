import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "Financial Wellness Lab", description: "Synthetic financial wellness workshop" };
export default function Layout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
