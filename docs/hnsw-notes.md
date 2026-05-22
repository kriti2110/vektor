# HNSW: study notes before you implement

> Read this BEFORE opening `vektor/index/hnsw.py`. The goal is to understand the algorithm well enough that you can explain it from memory to a Google interviewer.

## The one-sentence summary

HNSW is a multi-layer graph where upper layers are sparse (long jumps) and lower layers are dense (fine-grained neighbors). Search starts at the top, greedy-walks down through layers, refining as it descends. Insert assigns each new point a max level via a geometric distribution and connects it to M nearest neighbors at each layer up to its max.

## Why the layers?

- **Top layers (sparse).** Few nodes, long-range edges. Like an express train — get close to the answer cheaply.
- **Bottom layer (dense).** Every point lives here. Like a local train — refine to the exact answer.
- The combination gives **O(log N)** expected search time vs O(N) for flat.

## The key parameters

| Param | Meaning | Typical |
|-------|---------|---------|
| `M` | Max neighbors per node, per layer (except layer 0 = `2M`) | 16 |
| `mL` | Level-assignment normalization. Bigger → flatter graph. `mL = 1/ln(M)` is the paper's recommendation | `1/ln(16) ≈ 0.36` |
| `ef_construction` | Beam width during insertion's neighbor search | 200 |
| `ef_search` | Beam width during query-time search at layer 0. Bigger → higher recall, slower | 50–200 |

**Interview gotcha:** know the difference between `ef_construction` and `ef_search`. Construction is paid once, at build time. Search is paid per query.

## Level assignment

```python
def assign_level(self, mL: float) -> int:
    return int(-math.log(random.random()) * mL)
```

This is a geometric distribution. Most points land at level 0; very few at high levels. Specifically, the probability of landing at level `l` is `exp(-l/mL) * (1 - exp(-1/mL))`. With `mL = 1/ln(M)` and `M=16`, expect roughly:
- Level 0: ~94% of points
- Level 1: ~6%
- Level 2: ~0.4%
- Level 3: ~0.02%

Each layer's expected size is `1/M` of the layer below — that's where the log-N comes from.

## Search at one layer (greedy beam search)

Given an entry point and a query:

```
candidates = priority_queue(by distance, min-heap, capacity ef)
visited = {entry_point}
candidates.push(entry_point)
results = priority_queue(by distance, max-heap, capacity ef)
results.push(entry_point)

while candidates not empty:
    c = candidates.pop()  # closest unvisited
    f = results.peek()    # furthest in current results
    if dist(c) > dist(f): break  # can't improve

    for neighbor in c.neighbors:
        if neighbor in visited: continue
        visited.add(neighbor)
        if dist(neighbor) < dist(f) or len(results) < ef:
            candidates.push(neighbor)
            results.push(neighbor)
            if len(results) > ef: results.pop()  # evict furthest

return results sorted by distance
```

The early termination (`if dist(c) > dist(f): break`) is what makes this fast. Once the closest candidate is farther than the furthest result, no neighbor of that candidate can be closer either (by triangle inequality intuition, though not strictly guaranteed — this is why HNSW is *approximate*).

## Full search (across layers)

```
ep = self.entry_point
for layer in range(self.max_level, 0, -1):  # top down to layer 1
    ep = self.search_layer(query, ep, ef=1, layer)[0]  # greedy, ef=1

# At layer 0, do the real work
return self.search_layer(query, ep, ef=ef_search, layer=0)[:k]
```

The upper-layer searches use `ef=1` (pure greedy) because we just need to position ourselves near the answer. Only the bottom layer does the beam search.

## Insert

```
new_level = assign_level(mL)
ep = self.entry_point

# Descend without inserting until we reach new_level + 1
for layer in range(self.max_level, new_level, -1):
    ep = self.search_layer(q, ep, ef=1, layer)[0]

# From new_level downward, insert
for layer in range(min(new_level, self.max_level), -1, -1):
    M = self.M if layer > 0 else self.M * 2  # layer 0 gets 2M
    candidates = self.search_layer(q, ep, ef=ef_construction, layer)
    neighbors = self.select_neighbors_heuristic(q, candidates, M, layer)
    self.connect(new_node, neighbors, layer)
    # Bidirectional: also prune existing neighbors if they exceed M
    for n in neighbors:
        if len(n.neighbors_at(layer)) > M:
            n.neighbors_at(layer) = self.select_neighbors_heuristic(
                n, n.neighbors_at(layer), M, layer
            )
    ep = candidates[0]  # closest becomes entry for next layer down

if new_level > self.max_level:
    self.max_level = new_level
    self.entry_point = new_node
```

## The neighbor selection heuristic

This is what makes HNSW better than naive NSW. Instead of "pick the M nearest", you pick M neighbors that are both close AND diverse:

```
def select_neighbors_heuristic(q, candidates, M):
    # candidates sorted ascending by distance to q
    result = []
    for c in candidates:
        if len(result) >= M: break
        # only add c if it's closer to q than to any already-selected neighbor
        if all(dist(c, q) < dist(c, r) for r in result):
            result.append(c)
    return result
```

The intuition: if `c` is closer to some already-selected `r` than to the query, then `r` is a better path to `c`, and adding `c` directly doesn't add new graph structure. This prevents clustering of edges in one direction.

The Malkov paper has a more elaborate version with `extendCandidates` and `keepPrunedConnections`. Read §4 of the paper.

## Common bugs (avoid these)

1. **Off-by-one in level loops.** The descent loop goes `max_level → new_level+1` (inclusive of `new_level+1`, exclusive of `new_level`). The insert loop goes `min(new_level, max_level) → 0` inclusive.
2. **Not making neighbor links bidirectional.** When you connect new node `q` to neighbor `n`, you also have to add `q` to `n`'s neighbor list (and possibly evict one of `n`'s old neighbors via the heuristic).
3. **Forgetting to update `max_level` and `entry_point`** when the new node's level exceeds the current max.
4. **Visited set per query, not global.** Don't accumulate visited across searches — reset per call.
5. **Using `ef=1` at the wrong layer.** Greedy at upper layers, beam at layer 0.

## How to test your implementation

`tests/test_hnsw.py` will give you:
- A small graph fixture with known shortest paths, so you can assert greedy walks return correct results.
- A 10k-doc benchmark that asserts your recall@10 is within 2% of `FlatIndex`.
- A 100k-doc benchmark that asserts your query latency is < 5ms (single-threaded, after warmup).

Run with: `pytest tests/test_hnsw.py -v`.

## Recommended reading order

1. This file (you're here)
2. Pinecone's visual HNSW explainer — https://www.pinecone.io/learn/series/faiss/hnsw/
3. The original paper, especially Algorithm 1-5 — https://arxiv.org/abs/1603.09320
4. The `hnswlib` C++ source (only AFTER you've drafted your own) to compare design choices

## Interview questions you'll be ready for

- "Walk me through how a query is processed in HNSW."
- "Why a geometric distribution for level assignment?"
- "What's `ef_construction` vs `ef_search`?"
- "Why is the neighbor selection heuristic better than top-M?"
- "How would you make HNSW support updates/deletes? (It's hard — discuss why.)"
- "How would you shard HNSW across machines? (Hash partitioning + scatter-gather.)"
- "What's the recall/latency tradeoff curve look like as you sweep `ef_search`?"

If you can answer all of these in your own words, you've internalized the algorithm.
