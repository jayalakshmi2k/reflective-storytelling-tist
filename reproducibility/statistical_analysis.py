import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import wilcoxon, rankdata, binomtest

def getargs():
 p = argparse.ArgumentParser()
 p.add_argument("--gold-a", required=True)
 p.add_argument("--gold-b", required=True)
 p.add_argument("--output-dir", default="results/statistics")
 return p.parse_args()

def findcol(df, names):
 for x in names:
  if x in df.columns:
   return x
 raise ValueError("column not found " + str(names))

def loadfile(name):
 d = pd.read_csv(name)
 persona = findcol(d, ["persona", "Persona", "persona_id"])
 story = findcol(d, ["story", "Story", "story_id", "narrative", "narrative_id"])
 label = findcol(d, ["label", "gold_label", "Label"])
 d[label] = d[label].astype(str).str.strip().str.upper()
 return d, persona, story, label

def storycounts(d, persona, story, label):
 rows = []
 for k, x in d.groupby([persona, story], sort=False):
  labels = x[label]
  claims = (labels == "CLAIM").sum()
  premises = (labels == "PREMISE").sum()
  none = (labels == "NONE").sum()
  rows.append({
   "persona": k[0],
   "story": k[1],
   "claims": claims,
   "premises": premises,
   "none": none,
   "argumentative": claims + premises,
   "claim_premise": 1 if claims > 0 and premises > 0 else 0
  })
 return pd.DataFrame(rows)

def bootstrap(x, y, n=200000):
 rng = np.random.default_rng(42)
 diff = np.array(x, dtype=float) - np.array(y, dtype=float)
 vals = np.empty(n)
 for i in range(n):
  s = rng.integers(0, len(diff), len(diff))
  vals[i] = diff[s].mean()
 return np.quantile(vals, [.025, .975])

def rb(x, y):
 d = np.array(x, dtype=float) - np.array(y, dtype=float)
 d = d[d != 0]
 if len(d) == 0:
  return 0
 r = rankdata(abs(d), method="average")
 pos = r[d > 0].sum()
 neg = r[d < 0].sum()
 return (pos - neg) / (pos + neg)

def numeric_test(m, col):
 x = m[col + "_A"].to_numpy()
 y = m[col + "_B"].to_numpy()
 try:
  w, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
 except ValueError:
  w, p = 0.0, 1.0
 lo, hi = bootstrap(x, y)
 return {
  "outcome": col,
  "n": len(x),
  "mean_A": x.mean(),
  "mean_B": y.mean(),
  "median_A": np.median(x),
  "median_B": np.median(y),
  "mean_difference": (x - y).mean(),
  "ci_low": lo,
  "ci_high": hi,
  "wilcoxon_W": w,
  "p_value": p,
  "rank_biserial": rb(x, y)
 }

def binary_test(m, col):
 x = m[col + "_A"].astype(int)
 y = m[col + "_B"].astype(int)
 onlya = ((x == 1) & (y == 0)).sum()
 onlyb = ((x == 0) & (y == 1)).sum()
 n = onlya + onlyb
 if n == 0:
  p = 1.0
 else:
  p = binomtest(min(onlya, onlyb), n, .5, alternative="two-sided").pvalue
 return {
  "outcome": col,
  "n": len(m),
  "A_yes": int(x.sum()),
  "B_yes": int(y.sum()),
  "A_only": int(onlya),
  "B_only": int(onlyb),
  "risk_difference": x.mean() - y.mean(),
  "p_value": p
 }

def main():
 a = getargs()
 out = Path(a.output_dir)
 out.mkdir(parents=True, exist_ok=True)

 A, pa, sa, la = loadfile(a.gold_a)
 B, pb, sb, lb = loadfile(a.gold_b)

 A = storycounts(A, pa, sa, la)
 B = storycounts(B, pb, sb, lb)

 m = A.merge(B, on=["persona", "story"], suffixes=("_A", "_B"))

 if len(m) != 40:
  raise ValueError("Expected 40 matched narratives, found " + str(len(m)))

 print("matched narratives", len(m))

 results = []
 for col in ["premises", "claims", "argumentative", "none"]:
  r = numeric_test(m, col)
  results.append(r)
  print("\n" + col)
  print(r)

 b = binary_test(m, "claim_premise")
 print("\nclaim and premise present")
 print(b)

 pd.DataFrame(results).to_csv(out / "wilcoxon_results.csv", index=False)
 pd.DataFrame([b]).to_csv(out / "mcnemar_claim_premise.csv", index=False)
 m.to_csv(out / "matched_narrative_counts.csv", index=False)

 print("\nsaved in", out)

if __name__ == "__main__":
 main()
