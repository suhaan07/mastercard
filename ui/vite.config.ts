import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bind the loopback address explicitly. Vite's default `localhost` resolved
    // to an interface that http://127.0.0.1:5173 could not reach on Windows,
    // and 127.0.0.1 is the URL run.ps1 prints and the demo script tells you to
    // open.
    host: "127.0.0.1",
    strictPort: true,
    proxy: {
      // The console talks to the gateway, never straight to the scorer --
      // the same path a merchant integration would take.
      // run.ps1 takes -GatewayPort, so the proxy has to follow it rather than
      // pin 8080; otherwise every /api call 404s on a non-default port.
      "/api": {
        target: `http://127.0.0.1:${process.env.GATEWAY_PORT ?? 8080}`,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
