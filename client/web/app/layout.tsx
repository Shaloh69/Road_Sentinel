import "@/styles/globals.css";
import { Metadata, Viewport } from "next";
import clsx from "clsx";

import { Providers } from "./providers";

import { siteConfig } from "@/config/site";
import { fontHeading, fontMono, fontSans } from "@/config/fonts";
import { Sidebar } from "@/components/sidebar";
import { AnimatedBackground } from "@/components/animated-background";
import { PageTransition } from "@/components/page-transition";

export const metadata: Metadata = {
  title: {
    default: siteConfig.name,
    template: `%s - ${siteConfig.name}`,
  },
  description: siteConfig.description,
  icons: {
    icon: "/favicon.ico",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f5f7" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0e14" },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html suppressHydrationWarning lang="en">
      <head />
      <body
        className={clsx(
          "min-h-screen text-foreground font-sans antialiased",
          fontSans.variable,
          fontHeading.variable,
          fontMono.variable,
        )}
      >
        <Providers themeProps={{ attribute: "class", defaultTheme: "dark" }}>
          <AnimatedBackground />
          <Sidebar />
          <div className="ml-64 min-h-screen relative">
            <main className="w-full">
              <PageTransition>{children}</PageTransition>
            </main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
