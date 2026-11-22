import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";
import { fillTokens } from "./scripts/tokens.mjs";

const DATA = new URL("./src/data.json", import.meta.url);

/** Fill {{path|format}} tokens in index.html from src/data.json, so no number is typed by hand. */
function injectNumbers(): Plugin {
  return {
    name: "inject-numbers",
    transformIndexHtml: {
      order: "pre",
      handler: (html) => fillTokens(html, JSON.parse(readFileSync(DATA, "utf8"))),
    },
  };
}

export default defineConfig({
  plugins: [injectNumbers()],
});
