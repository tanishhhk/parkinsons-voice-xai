import { dp2, int } from "../format";
import { linear, svgEl, svgText, svgTitle, ticks } from "./svg";

export interface PersonRow {
  subject: string;
  truth: number;
  prob: number;
  outcome: string;
}

const ROW_H = 54;
const TOP = 34;
const PAD_L = 56;
const PAD_R = 24;
const AXIS_H = 52;

/** Plain word for each confusion outcome. */
const WORD: Record<string, string> = {
  "true positive": "caught",
  "true negative": "cleared",
  "false positive": "false alarm",
  "false negative": "missed",
};

/**
 * Four held-out people on one 0-1 risk scale with the alarm threshold. Parkinson's is a tulip
 * diamond, healthy a calm circle; every outcome is also named in text beside its marker.
 */
export function renderPeople(el: HTMLElement, people: PersonRow[], threshold: number): void {
  let width = 0;

  const draw = () => {
    width = Math.round(el.clientWidth) || 320;
    el.replaceChildren();
    el.setAttribute("role", "group");

    const keys = document.createElement("p");
    keys.className = "chart-keys";
    keys.innerHTML =
      '<span class="ck ck-pd"><i></i>Has Parkinson\'s</span><span class="ck ck-healthy"><i></i>Healthy</span>';
    el.appendChild(keys);

    const W = Math.max(280, width);
    const x = linear([0, 1], [PAD_L, W - PAD_R]);
    const axisY = TOP + people.length * ROW_H;
    const H = axisY + AXIS_H;
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      class: "people-svg",
      role: "img",
      "aria-label": `Risk scores of ${int(people.length)} held-out people against an alarm threshold of ${dp2(threshold)}. ${people
        .map((p) => `${p.subject} ${WORD[p.outcome] ?? p.outcome}, scored ${dp2(p.prob)}`)
        .join("; ")}.`,
    });
    el.appendChild(svg);

    for (const t of ticks(0, 1, 0.25)) {
      svgEl("line", { x1: x(t), x2: x(t), y1: TOP - 8, y2: axisY, class: "grid" }, svg);
      svgText(svg, dp2(t), { x: x(t), y: axisY + 16, class: "tick", "text-anchor": "middle" });
    }
    svgText(svg, "Risk score from 0 to 1, held-out people only", {
      x: 0,
      y: axisY + 46,
      class: "axis-title",
    });

    people.forEach((p, i) => {
      const cy = TOP + i * ROW_H + ROW_H / 2;
      const pd = p.truth === 1;
      const word = WORD[p.outcome] ?? p.outcome;
      const flagged = p.prob >= threshold;
      const g = svgEl(
        "g",
        { class: `person ${pd ? "is-pd" : "is-healthy"}`, "data-subject": p.subject, "data-outcome": word },
        svg,
      );
      svgTitle(
        g,
        `${p.subject}: ${pd ? "Parkinson's" : "healthy"}, scored ${dp2(p.prob)}, ${flagged ? "flagged" : "not flagged"}: ${word}.`,
      );
      svgEl("line", { x1: x(0), x2: x(1), y1: cy, y2: cy, class: "lane" }, g);
      const mx = x(p.prob);
      if (pd) {
        svgEl("path", { d: `M${mx} ${cy - 10} L${mx + 10} ${cy} L${mx} ${cy + 10} L${mx - 10} ${cy} Z`, class: "mark mark-pd" }, g);
      } else {
        svgEl("circle", { cx: mx, cy, r: 8.5, class: "mark mark-healthy" }, g);
      }
      // The label sits on the side of the marker facing away from the threshold, so it never crosses it.
      // Flagged people normally label to the right; near the right edge the label flips to the left.
      const right = flagged && mx + 16 + 96 <= W;
      const lx = right ? mx + 16 : mx - 16;
      const anchor = right ? "start" : "end";
      svgText(g, p.subject, { x: lx, y: cy - 3, class: "p-name", "text-anchor": anchor });
      svgText(g, `${word} ${dp2(p.prob)}`, { x: lx, y: cy + 12, class: "p-word", "text-anchor": anchor });
    });

    svgEl("line", { x1: x(threshold), x2: x(threshold), y1: TOP - 14, y2: axisY, class: "chance" }, svg);
    svgText(svg, "not flagged", { x: x(threshold) - 8, y: TOP - 18, class: "zone", "text-anchor": "end" });
    svgText(svg, "flagged", { x: x(threshold) + 8, y: TOP - 18, class: "zone" });
    svgText(svg, `threshold ${dp2(threshold)}`, {
      x: x(threshold),
      y: axisY + 32,
      class: "chance-label",
      "text-anchor": "middle",
    });
  };

  draw();
  new ResizeObserver(() => {
    const w = Math.round(el.clientWidth);
    if (w > 0 && w !== width) draw();
  }).observe(el);
}
