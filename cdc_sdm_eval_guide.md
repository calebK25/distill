
# Cross-Document Compression & Single‑Doc Mode — Implementation Guide

This document specifies how to implement **Cross‑Document Compression (CDC)** and a **Single‑Document Mode (SDM)** for the Context Compressor, plus an evaluation suite, **fine‑tuning recipes**, and acceptance metrics.

---

## 1) Scope

- **Feature A — Cross‑Document Compression (CDC):** Select and trim evidence across many documents under a token budget **B**, maximizing coverage and minimizing redundancy while preserving citation traceability.
- **Feature B — Single‑Document Mode (SDM):** When a query primarily targets one document, compress across its sections/pages only, with section caps and table/prose fusion.
- **Fine‑Tuning:** Train/improve scorers (bi‑encoder & cross‑encoder) and, optionally, a budgeted selector using oracle labels.
- **Evaluation:** Token reduction, answer quality deltas, coverage, redundancy, latency, and routing quality.

Assumes existing components: retrieval → candidates, token counter, sentence splitter, optional cross‑encoder.

---

## 2) Data Model

**Candidate span** (sentence or short paragraph):
```json
{
  "id": "c_001",
  "doc_id": "d_01",
  "section": "Results",
  "page": 14,
  "text": "Q3 revenue grew 12% YoY to $1.24B …",
  "tokens": 29,
  "bm25": 7.2,
  "dense_sim": 0.81,
  "embedding": [ ... ]  // optional float32
}
```

**Compressor request:**
```json
{
  "q": "What were the key results in Q3 2024?",
  "B": 1500,
  "candidates": [ ... ],
  "params": {
    "fusion_weights": {"dense": 0.7, "bm25": 0.3},
    "lambda": 0.7,
    "section_cap": 2,
    "doc_cap": 6,
    "topM": 200,
    "auto_router": true,
    "use_reranker": false
  }
}
```

**Response:**
```json
{
  "context": "…compressed spans…",
  "mapping": [
    {"id":"c_001","doc_id":"d_01","section":"Results","page":14,"tokens":29,"trimmed":false}
  ],
  "stats": {
    "mode": "cross_doc|single_doc",
    "budget":1500,"used":980,"saved_vs_pool":4320,
    "lambda":0.7,"fusion_weights":{"dense":0.7,"bm25":0.3},
    "section_cap":2,"doc_cap":6,
    "router_score":{"top1_doc_frac":0.86, "entropy":0.42},
    "low_context": false
  }
}
```

---

## 3) Feature A — Cross‑Document Compression (CDC)

### 3.1 Rank Fusion
Compute fused relevance per candidate and keep top‑M for the MMR pool.
```python
def fuse_scores(cands, w_dense=0.7, w_bm25=0.3, M=200):
    import numpy as np
    d = np.array([c["dense_sim"] for c in cands], float)
    b = np.array([c["bm25"] for c in cands], float)
    zd = (d - d.mean()) / (d.std() + 1e-9)
    zb = (b - b.mean()) / (b.std() + 1e-9)
    for c, s in zip(cands, w_dense*zd + w_bm25*zb):
        c["fusion"] = float(s)
    return sorted(cands, key=lambda x: x["fusion"], reverse=True)[:M]
```

### 3.2 Global MMR (budget‑aware, cross‑doc)
Greedy selection to balance query relevance and novelty; enforce **doc_cap** and **section_cap**.
```python
def select_mmr_crossdoc(query_vec, pool, B, lam=0.7, section_cap=2, doc_cap=6):
    import numpy as np
    S, used, per_section, per_doc = [], 0, {}, {}
    def cos(a,b): return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
    while pool and used < B:
        def mmr_score(c):
            sim_q = cos(query_vec, c["embedding"])
            if not S: return lam*sim_q
            max_sim = max(cos(c["embedding"], s["embedding"]) for s in S)
            return lam*sim_q - (1-lam)*max_sim
        pool.sort(key=mmr_score, reverse=True)
        c = pool.pop(0)
        if per_doc.get(c["doc_id"], 0) >= doc_cap:
            continue
        if per_section.get((c["doc_id"], c["section"]), 0) >= section_cap:
            continue
        if used + c["tokens"] > B * 1.15:  # soft overflow before trimming
            c = trim_in_chunk(c, query_vec, target_tokens=max(0, B - used))
            if c["tokens"] == 0: continue
        S.append(c)
        used += c["tokens"]
        per_doc[c["doc_id"]] = per_doc.get(c["doc_id"], 0) + 1
        key = (c["doc_id"], c["section"])
        per_section[key] = per_section.get(key, 0) + 1
    return S, used
```

### 3.3 In‑Chunk Trimming (sentence level)
Rank sentences by query‑sentence cosine, plus anchor hits (numbers/entities). Keep order; maintain citation ID.
```python
def trim_in_chunk(c, query_vec, target_tokens):
    sents = split_sentences(c["text"])
    scored = []
    for s in sents:
        sv = embed_sentence(s)
        score = 0.8*cos(query_vec, sv) + 0.2*anchor_bonus(s)  # anchors: numbers, entities
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    kept, tok = [], 0
    for _, s in scored:
        t = count_tokens(s)
        if tok + t > target_tokens: 
            continue
        kept.append(s); tok += t
        if tok >= target_tokens: 
            break
    if tok == 0: 
        return {"id": c["id"], "doc_id": c["doc_id"], "section": c["section"],
                "page": c["page"], "text":"", "tokens":0, "embedding": c["embedding"]}
    out = " ".join(sorted(kept, key=lambda s: sents.index(s)))
    return {**c, "text": out, "tokens": count_tokens(out), "trimmed": True}
```

---

## 4) Feature B — Single‑Document Mode (SDM) with Auto‑Router

### 4.1 Router
Decide mode by concentration of top retrieval among documents.
```python
def route_mode(candidates, threshold=0.8):
    from collections import Counter
    top = candidates[:50]  # by fusion or raw retrieval
    freq = Counter([c["doc_id"] for c in top])
    top1_doc, count = max(freq.items(), key=lambda kv: kv[1])
    frac = count / max(1, len(top))
    # entropy for tie‑break
    import math
    p = [v/len(top) for v in freq.values()]
    H = -sum(x*math.log(x+1e-9) for x in p)
    return ("single_doc", top1_doc, {"top1_doc_frac": frac, "entropy": H}) if frac >= threshold else ("cross_doc", None, {"top1_doc_frac": frac, "entropy": H})
```

### 4.2 SDM selection
Filter pool to the routed `doc_id`, then run the same fusion → MMR with a per‑section cap. Keep **doc_cap** very high in SDM.
```python
def compress_single_doc(q_vec, candidates, doc_id, B, lam=0.7, section_cap=2):
    pool = [c for c in candidates if c["doc_id"] == doc_id]
    S, used = select_mmr_crossdoc(q_vec, pool, B, lam, section_cap, doc_cap=10**6)
    return S, used
```

### 4.3 Putting it together
```python
def compress(q_vec, candidates, B, params):
    pool = fuse_scores(candidates, params["fusion_weights"]["dense"], params["fusion_weights"]["bm25"], params["topM"])
    mode, doc_id, rstats = route_mode(pool) if params.get("auto_router", True) else ("cross_doc", None, {})
    if mode == "single_doc":
        S, used = compress_single_doc(q_vec, pool, doc_id, B, params["lambda"], params["section_cap"])
    else:
        S, used = select_mmr_crossdoc(q_vec, pool, B, params["lambda"], params["section_cap"], params["doc_cap"])
    context = "\n\n".join(c["text"] for c in S)
    mapping = [{"id":c["id"], "doc_id":c["doc_id"], "section":c["section"], "page":c["page"], "tokens":c["tokens"], "trimmed":c.get("trimmed", False)} for c in S]
    stats = {"mode": mode, "budget": B, "used": used, "lambda": params["lambda"],
             "fusion_weights": params["fusion_weights"], "section_cap": params["section_cap"],
             "doc_cap": params["doc_cap"], "router_score": rstats, "low_context": used < 0.3*B}
    return {"context": context, "mapping": mapping, "stats": stats}
```

---

## 5) Fine‑Tuning (Detailed)

### 5.1 Goals
- Improve **query→sentence** relevance and novelty scoring to preserve QA accuracy at smaller budgets **B**.
- Optional: learn a **budgeted selector** that directly optimizes coverage under **B**.

### 5.2 Datasets & Labels

**A) From QA with citations (preferred):**
- For each example `(q, answer, cited_spans)`:
  - **Positives:** sentences that contain/support cited facts. Use a lightweight NLI to validate support (entail ≥ τ) and keep numeric‑consistent spans.
  - **Hard negatives:** (i) near sentences that *don’t* support the claim, (ii) high‑overlap sentences from other docs that lack anchors.
  - Build an **oracle pack**: the smallest set of sentences that covers **answer anchors** (entities, numbers+units, keyphrases) under budget **B** using greedy set‑cover.

**B) Self‑supervision (when no citations):**
- Run current compressor, then (i) use **answer anchors** to find supporting sentences, (ii) mine negatives from top‑K that were not chosen but have high lexical overlap, (iii) treat compressor‑kept sentences as weak positives.

**Schema (`train.jsonl`):**
```json
{
  "qid": "q_001",
  "q": "What are Q3 2024 key results?",
  "sentences": [
    {"doc_id":"d1","sid":"s1","text":"Revenue grew 12% to $1.24B", "tokens":18},
    {"doc_id":"d1","sid":"s2","text":"Operating margin was 5.0%.", "tokens":8}
  ],
  "oracle_sids": ["s1","s2"],
  "anchors": {"numbers":[["12","%"],["1.24","B"],["5.0","%"]], "entities":["Revenue","Operating margin"]}
}
```

### 5.3 Track A — Fine‑tune the Scorers (recommended)

**Bi‑encoder (embeddings) — contrastive**
- Model: `BAAI/bge-large-en-v1.5` (English) or `bge-m3` (multilingual).
- Construct pairs `(q, s_pos)` vs `(q, s_neg)`; train with InfoNCE:
  \[
  \mathcal{L}=-\log \frac{\exp(\mathrm{sim}(q,s^+)/\tau)}{\sum_{s\in\{s^+,\mathcal{N}\}}\exp(\mathrm{sim}(q,s)/\tau)}
  \]
- **Hard negatives:** mine from (i) same section neighbors, (ii) other docs with shared entities but mismatched numbers.

**Cross‑encoder reranker — pairwise/listwise**
- Model: `BAAI/bge-reranker-large`.
- Build triplets `(q, s_pos, s_neg)` where current ranking puts `s_neg` above `s_pos` (hard negatives).
- Loss: margin ranking: \(\max(0, m - f(q,s^+) + f(q,s^-))\), `m=0.1–0.3`.

**Training tips (4080)**
- Batch sizes: bi‑encoder 128 fp16; cross‑encoder 32.
- LR: `2e-5` with cosine decay, warmup 500 steps.
- Epochs: 1–3 (bi‑encoder), 1–2 (cross‑encoder) with early stopping on validation `P@k`.
- Mixed precision & gradient accumulation for large batches.

**Sentence‑Transformers quickstart (bi‑encoder):**
```python
from sentence_transformers import SentenceTransformer, losses, InputExample, SentencesDataset
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
train = [InputExample(texts=[ex["q"], pos["text"]]) for ex in positives] + \
        [InputExample(texts=[ex["q"], neg["text"]]) for ex in negatives]
loader = DataLoader(train, shuffle=True, batch_size=128)
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, train_loss)], epochs=2, warmup_steps=500, use_amp=True)
model.save("models/bge-large-en-ctxc")
```

**Cross‑encoder quickstart:**
```python
from sentence_transformers import CrossEncoder, InputExample
train = [InputExample(texts=[q, pos, neg], label=1.0) for (q,pos,neg) in triplets]
model = CrossEncoder("BAAI/bge-reranker-large", num_labels=1)
model.fit(train, epochs=1, batch_size=32, warmup_steps=500, output_path="models/reranker-ctxc")
```

**Integration**
- Swap the bi‑encoder into: fusion (dense), MMR, sentence trimming (`embed_sentence`).
- Use the reranker only on final ~30 candidates before budget fit.

### 5.4 Track B — Budgeted Selector (advanced)

**Oracle creation (greedy set‑cover)**
- **Anchors:** extract entities (NER), keyphrases, numbers+units (normalize with `pint`), dates.
- **Gain(c):** number of *new* anchors covered by candidate sentence `c`.
- **Penalty(c):** redundancy vs. already selected (max cosine similarity).
- **Objective:** pick sentences maximizing `Gain − α·Penalty` under token budget **B**.

**Modeling options**
- **Knapsack scorer:** predict a value `v_i` per sentence; solve knapsack (weight=tokens). Train with listwise loss toward oracle distribution (softmax over oracle picks).
- **Pointer network / decoder:** autoregressively output sentence indices; train to oracle sequence, add length penalty to respect **B**.
- **RL (REINFORCE):** reward = coverage − redundancy − over‑budget penalty. Use a strong baseline (e.g., knapsack).

**When to attempt:** only if Track A plateaus and you need further token cuts without quality loss.

### 5.5 Validation & Calibration
- **Dev split:** hold out 10–20% of queries; compute `P@k` for sentence retrieval and ΔEM/ΔP@5 end‑to‑end.
- **Ablations:** old vs. new bi‑encoder; reranker on/off; different **B**.
- **Early stopping:** on ΔP@5 (end‑to‑end) and anchor coverage.
- **Checkpoint policy:** store `model_hash` and `params.json`; write a short “Model Card” noting data sources, domains, and known limits.

---

## 6) Evaluation Suite

### 6.1 Metrics (report mean ± 95% CI)
- **Token reduction (%):** `(tokens_in_pool - tokens_out)/tokens_in_pool`.
- **Answer quality deltas:** `ΔEM`, `ΔP@5` vs. no‑compression baseline.
- **Coverage:** fraction of **answer anchors** present in the compressed context.
- **Redundancy:** mean pairwise cosine among selected sentences (lower is better).
- **Latency:** p50/p95 compressor time (fusion, MMR, trimming, total), with and without reranker.
- **Mode routing:** % queries routed to SDM, and their quality/latency vs. CDC.
- **Second‑pass rate:** % “insufficient context”; aim ≤ 2–3% at `B=1500`.

### 6.2 Acceptance criteria
- Token reduction: **50–70%** with **ΔEM/ΔP@5 ≥ −1 pt** overall.
- Latency: p95 ≤ **40 ms** for `N≤200` (no reranker); ≤ **120 ms** with reranker on shortlist.
- Coverage: ≥ **0.9** of anchors for single‑doc factual questions.
- Redundancy: ≤ baseline by **30%**.

### 6.3 Harness outline
```python
def evaluate(dataset, compressor, B=1500):
    rows = []
    for ex in dataset:  # ex = {q, q_vec, candidates, anchors, gold_answer, baseline_context}
        t0 = now()
        out = compressor(ex["q_vec"], ex["candidates"], B, params)
        t_ms = ms_since(t0)
        cov = anchor_coverage(ex["anchors"], out["context"])
        red = avg_pairwise_cosine([embed_sentence(c["text"]) for c in out["mapping"]])
        em, p5 = score_answer(ex["q"], out["context"], ex["gold_answer"])
        base_em, base_p5 = score_answer(ex["q"], ex["baseline_context"], ex["gold_answer"])
        rows.append({"cov":cov,"red":red,"em":em,"p5":p5,"base_em":base_em,"base_p5":base_p5,"lat_ms":t_ms,"mode":out["stats"]["mode"]})
    return summarize(rows)
```

---

## 7) Defaults (4080 + CPU)
- **Embeddings:** `BAAI/bge-large-en-v1.5` (fp16, batch 128).
- **Reranker:** `BAAI/bge-reranker-large` on final 30 (batch 32) — optional.
- **Params:** `topM=200`, `lambda=0.7`, `section_cap=2`, `doc_cap=6`, `B=1500`.
- **Token counter:** match downstream LLM tokenizer.

---

## 8) Tests
- Unit: fusion math, MMR selection, budget fit, section/doc caps, determinism (seed fixed).
- Property: trimming preserves sentence order; never exceed **B**.
- Golden: stable selection on 10–20 curated queries.
- Perf: synthetic `N=200`; assert p95 < 40 ms (no reranker).

---

## 9) Deliverables
- `context_compressor/` module with CDC + SDM + router
- FastAPI `POST /compress` exposing `mode`, `mapping`, `stats`
- `fine_tune/` scripts for bi‑encoder and cross‑encoder
- `eval/` harness + 50‑query smoke dataset
- README with metrics, defaults, and example requests
