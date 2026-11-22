export type PersonRow = {
  repeat: number;
  subject: string;
  truth: 0 | 1;
  prob: number;
  threshold: number;
};

export type OperatingPoint = {
  sensitivity: number | null;
  specificity: number | null;
  auc: number;
  counts: { caught: number; missed: number; cleared: number; falseAlarms: number };
};

function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}

function byRepeat(rows: PersonRow[]): PersonRow[][] {
  const groups = new Map<number, PersonRow[]>();
  for (const r of rows) {
    const g = groups.get(r.repeat);
    if (g) g.push(r);
    else groups.set(r.repeat, [r]);
  }
  return [...groups.values()];
}

/** Rank-based AUC for one set of rows (ties count half); NaN if a class is missing. */
function aucOne(rows: PersonRow[]): number {
  const pos = rows.filter((r) => r.truth === 1);
  const neg = rows.filter((r) => r.truth === 0);
  if (!pos.length || !neg.length) return NaN;
  let wins = 0;
  for (const p of pos) {
    for (const n of neg) {
      if (p.prob > n.prob) wins += 1;
      else if (p.prob === n.prob) wins += 0.5;
    }
  }
  return wins / (pos.length * neg.length);
}

/** Mean of the per-repeat AUCs, skipping repeats with one class. NaN if none. */
export function auc(rows: PersonRow[]): number {
  const vals = byRepeat(rows).map(aucOne).filter((v) => !Number.isNaN(v));
  return mean(vals) ?? NaN;
}

/** Operating point at a threshold, or at each row's own frozen threshold. */
export function operatingPoint(
  persons: PersonRow[],
  threshold: number | "frozen",
): OperatingPoint {
  const sens: number[] = [];
  const spec: number[] = [];
  const tps: number[] = [];
  const fns: number[] = [];
  const tns: number[] = [];
  const fps: number[] = [];
  for (const g of byRepeat(persons)) {
    let tp = 0, fn = 0, tn = 0, fp = 0;
    for (const r of g) {
      const t = threshold === "frozen" ? r.threshold : threshold;
      const pred = r.prob >= t;
      if (r.truth === 1) pred ? tp++ : fn++;
      else pred ? fp++ : tn++;
    }
    if (tp + fn > 0) sens.push(tp / (tp + fn));
    if (tn + fp > 0) spec.push(tn / (tn + fp));
    tps.push(tp);
    fns.push(fn);
    tns.push(tn);
    fps.push(fp);
  }
  return {
    sensitivity: mean(sens),
    specificity: mean(spec),
    auc: auc(persons),
    counts: {
      caught: mean(tps) ?? 0,
      missed: mean(fns) ?? 0,
      cleared: mean(tns) ?? 0,
      falseAlarms: mean(fps) ?? 0,
    },
  };
}
