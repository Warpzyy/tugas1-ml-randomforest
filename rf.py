"""Random Forest (Gain Ratio / C4.5) dengan log langkah demi langkah."""
import math, random
from collections import Counter

def entropy(rows):
    n = len(rows)
    c = Counter(r["y"] for r in rows)
    return -sum(v / n * math.log2(v / n) for v in c.values() if v)

def evaluate(rows, fi):
    n, e0 = len(rows), entropy(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r["x"][fi], []).append(r)
    gain, split, info = e0, 0.0, []
    for v, g in sorted(groups.items()):
        p = len(g) / n
        gain -= p * entropy(g)
        split -= p * math.log2(p)
        info.append({"v": v, "n": len(g), "counts": dict(Counter(r["y"] for r in g)),
                     "ent": round(abs(entropy(g)), 6), "term": round(abs(-p * math.log2(p)), 6)})
    gr = gain / split if split else 0.0
    return {"gain": round(gain, 6), "split": round(split, 6), "gr": round(gr, 6)}, info, groups

def build(rows, avail, rng, m, name, F, steps):
    counts = Counter(r["y"] for r in rows)
    maj = counts.most_common(1)[0][0]
    if len(counts) == 1 or not avail:
        return {"leaf": maj, "n": len(rows), "counts": dict(counts)}
    cand = rng.sample(avail, min(m, len(avail)))
    res = {fi: evaluate(rows, fi) for fi in cand}
    best = max(cand, key=lambda f: res[f][0]["gr"])
    steps.append({"name": name, "n": len(rows), "counts": dict(counts),
                  "entropy": round(entropy(rows), 6), "chosen": F[best],
                  "cands": [{"feat": F[f], **res[f][0], "groups": res[f][1]} for f in cand]})
    node = {"feat": F[best], "fi": best, "maj": maj, "n": len(rows), "children": {}}
    rest = [a for a in avail if a != best]
    for v, g in sorted(res[best][2].items()):
        node["children"][v] = build(g, rest, rng, m, f"{F[best]} = {v}", F, steps)
    return node

def predict(node, x, F):
    path = []
    while "leaf" not in node:
        v = x[node["fi"]]
        path.append(f'{node["feat"]} = {v}')
        if v not in node["children"]:
            return node["maj"], path
        node = node["children"][v]
    return node["leaf"], path

def run_forest(p):
    F = p["features"]
    R = [{"id": i + 1, "x": r[:-1], "y": r[-1]} for i, r in enumerate(p["rows"])]
    n, m, target, out = len(R), int(p.get("m", 2)), p["target"], []
    for t in p["trees"]:
        s = str(t.get("seed", "")).strip()
        rng = random.Random(int(s) if s else None)
        raw = str(t.get("ids", "")).strip()
        ids = [int(i) for i in raw.replace(";", ",").split(",") if i.strip()] if raw else \
              [rng.randint(1, n) for _ in range(n)]
        if any(i < 1 or i > n for i in ids):
            raise ValueError("Row ID di luar rentang 1-%d" % n)
        sample = [R[i - 1] for i in ids]
        steps = []
        tree = build(sample, list(range(len(F))), rng, m, "Root", F, steps)
        label, path = predict(tree, target, F)
        out.append({"ids": ids, "oob": sorted(set(range(1, n + 1)) - set(ids)),
                    "sample": [{"id": i, "x": R[i - 1]["x"], "y": R[i - 1]["y"]} for i in ids],
                    "counts": dict(Counter(r["y"] for r in sample)), "steps": steps,
                    "tree": tree, "pred": label, "path": path})
    votes = Counter(o["pred"] for o in out)
    return {"trees": out, "votes": dict(votes), "final": votes.most_common(1)[0][0]}
