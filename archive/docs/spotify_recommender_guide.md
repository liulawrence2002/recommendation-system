# Building a State-of-the-Art Music Recommender (Spotify-Style)
### A practitioner's guide to the modern recommendation stack — from collaborative filtering to generative recommendation with Semantic IDs

> **Who this is for.** ML engineers who are comfortable with Python, PyTorch, embeddings, and neural nets, and want a faithful map of how production music recommenders are actually built in 2025–2026 — including the generative-retrieval frontier that Spotify, Google, and Meta are now shipping. Every major component comes with runnable starter code you can lift into a real project.
>
> **How to read it.** Sections 1–4 give you the mental model and the data foundations. Sections 5–9 build the *classic* (and still dominant) multi-stage stack: embeddings → retrieval → sequential modeling → ranking → re-ranking. Sections 10–11 cover the *frontier*: Semantic IDs, generative retrieval (TIGER/HSTU), and LLM-as-recommender. Sections 12–14 cover evaluation, serving, and a phased build roadmap. Section 15 is a curated reading list with the primary sources.

---

## Table of contents

1. The big picture: what a music recommender actually is
2. The two paradigms: multi-stage funnel vs. generative recommendation
3. Framing the music problem (implicit feedback, sequences, cold start, exploration)
4. Data and feature foundations
5. Representations I — collaborative and graph embeddings
6. Representations II — audio and multimodal content embeddings
7. Stage 1 — candidate retrieval with a two-tower model
8. Stage 2 — sequential modeling (SASRec / BERT4Rec)
9. Stage 3 — ranking (multi-task DLRM / DCN)
10. Stage 4 — re-ranking, diversity, calibration, and bandits
11. The frontier — generative recommendation and Semantic IDs
12. Cold start
13. Evaluation: offline, counterfactual, and online
14. Serving and system architecture
15. A concrete build roadmap
16. Reading list and primary sources

---

## 1. The big picture: what a music recommender actually is

A music recommender is not "an algorithm." It is a **system of models** that together answer one question millions of times per second: *given everything we know about this listener and this moment, what should we play or surface next?*

Three properties make music recommendation distinct from, say, e-commerce:

1. **Consumption is sequential and session-based.** What you want depends heavily on what you just played, the time of day, and your immediate context (workout, focus, sleep). The unit of value is often a *good next track* or a *good session*, not a one-off purchase.
2. **The catalog is enormous and constantly growing.** Tens of millions of tracks, plus podcasts and audiobooks, with tens of thousands of new items per day. You cannot score the whole catalog for every request, and you must handle brand-new items with no interaction history (cold start).
3. **Feedback is implicit and noisy.** Nobody hands you a 1–5 star rating. You infer preference from plays, skips, saves, completion rate, repeat listens, and dwell time — all of which are biased by what the system chose to show in the first place.

A useful way to internalize the field: **almost every production recommender, including Spotify's, is a funnel that trades off recall for precision as it narrows millions of candidates down to a handful of slots.** The frontier (Section 11) is changing *how* that funnel is implemented — replacing hand-assembled stages with a single generative model — but the funnel logic still governs the economics.

Spotify's own pipeline is described publicly as exactly this multi-stage pattern: candidate retrieval narrows millions of tracks to a few thousand, ranking scores those candidates, and re-ranking applies diversity and business constraints before presenting the final set. On the Home page specifically, the system that orders the "shelves" of cards is **BaRT** ("Bandits for Recommendations as Treatments"), which uses a contextual/multi-armed-bandit approach (an ε-greedy policy) to balance exploitation (familiar music) against exploration (discovery).

---

## 2. The two paradigms: multi-stage funnel vs. generative recommendation

You should hold both of these in your head at once. The first is what virtually everyone runs in production today. The second is what the leading labs are actively shipping and what "most state of the art" now points to.

### 2.1 The classic multi-stage funnel (the workhorse)

```
                ┌─────────────────────────────────────────────────────────┐
   ~50M items   │  STAGE 1: CANDIDATE RETRIEVAL (a.k.a. candidate gen)     │
  ───────────►  │  Cheap, high-recall. Many parallel "retrievers":         │
                │   • Two-tower ANN (user-embedding → top-K tracks)        │
                │   • Sequential model (next-item)                         │
                │   • Co-listening / item-item (i2i)                       │
                │   • Content / audio similarity (cold start)              │
                │   • Popularity / editorial / fresh                       │
                └───────────────┬─────────────────────────────────────────┘
                      ~1,000–10,000 candidates
                                │
                ┌───────────────▼─────────────────────────────────────────┐
   scoring      │  STAGE 2: RANKING                                        │
  ───────────►  │  Expensive, high-precision. One heavy model scores       │
                │  each (user, item) pair with rich cross features.        │
                │  Usually MULTI-TASK: P(play), P(skip), P(save),          │
                │  P(complete), predicted listen time…                     │
                └───────────────┬─────────────────────────────────────────┘
                       ~100–500 ranked items
                                │
                ┌───────────────▼─────────────────────────────────────────┐
   policy       │  STAGE 3: RE-RANKING / POLICY                            │
  ───────────►  │  Diversity (MMR/DPP), calibration to user's taste mix,   │
                │  freshness, fatigue, business rules, EXPLORATION         │
                │  (bandits, e.g. BaRT on Home).                           │
                └───────────────┬─────────────────────────────────────────┘
                         final slate (10–50)
```

Why a funnel? **Cost.** A two-tower retriever is built so the item side can be precomputed and indexed for approximate nearest-neighbor (ANN) search; serving a request is one user-embedding forward pass plus a sublinear ANN lookup. The ranker is far too expensive to run on 50M items, but perfectly affordable on a few thousand. The re-ranker is where product policy lives.

The key engineering constraint that shapes everything: **retrieval must allow late interaction.** User and item are encoded by separate towers that don't fuse until a final dot product, so item embeddings can be precomputed and indexed. The ranker, by contrast, is allowed *early* interaction — it can cross user and item features arbitrarily because it only runs on a small candidate set.

### 2.2 The generative paradigm (the frontier)

Instead of "embed everything, do ANN, then score," **generative recommendation reframes the whole problem as sequence generation**, exactly like a language model:

1. Give every item a short sequence of discrete tokens — a **Semantic ID** — derived from its embedding (so similar items share prefix tokens).
2. Represent a user as the **sequence of Semantic IDs** they've interacted with.
3. Train a Transformer to **autoregressively generate the Semantic ID of the next item**, token by token, the way an LLM predicts the next word.

This is the approach behind Google's **TIGER** (Transformer Index for Generative Recommenders), Meta's **HSTU / "Actions Speak Louder than Words"** generative recommenders (which demonstrated *recommendation scaling laws* — quality scales as a power law of compute up to GPT-3 scale), and Spotify's 2025 work on **Semantic IDs** and domain-adapting an open-weight **LLM to "speak Spotify."**

Why it matters: it can collapse retrieval + ranking into one model, generalizes to cold-start items (a new item gets a Semantic ID from its content embedding immediately), unifies search and recommendation, and — uniquely — *keeps improving as you scale parameters and data*, which classic ID-embedding DLRMs largely do not.

You do not have to choose. A pragmatic 2026 architecture often uses generative retrieval as **one powerful retriever inside the funnel**, with Semantic IDs as features in the ranker. Section 11 builds this out.

---

## 3. Framing the music problem

Before any modeling, nail the formalism — most real-world failures are framing failures.

### 3.1 Implicit feedback, not ratings

You observe interactions, not preferences. Convert raw events into a training signal carefully:

- **Positive signals:** play-to-completion, save/like, add-to-playlist, repeat listens, long dwell.
- **Negative / ambiguous:** **skips** (especially an early skip, < 30s — a strong negative), impressions with no play (weak negative), thumbs-down.
- **Confidence weighting.** The classic Hu–Koren–Volinsky insight (implicit ALS): treat the binary "interacted or not" as a preference, but attach a **confidence** that grows with how much evidence you have (e.g., number of plays, completion fraction). A track played 40 times is a much stronger positive than one played once.

A simple, effective label schema for music:

```python
def make_label(event):
    # returns (preference in {0,1}, confidence weight)
    if event.completion >= 0.9 or event.saved or event.added_to_playlist:
        return 1, 1.0 + 0.5 * event.play_count       # strong positive, grows w/ evidence
    if 0.3 <= event.completion < 0.9:
        return 1, 0.3                                  # weak positive
    if event.completion < 0.05 and event.was_user_initiated_skip:
        return 0, 1.0                                  # strong negative (early skip)
    return 0, 0.1                                       # impression / unknown
```

### 3.2 The selection-bias trap (this is the big one)

Your logs only contain feedback on items **the current system chose to show**. If you train naively on this data, you learn to imitate the existing system's blind spots — a feedback loop that collapses diversity over time. Two defenses you'll see throughout this guide:

- **Exploration** at serving time (bandits, ε-greedy, Section 10) so the logs keep sampling the catalog.
- **Counterfactual / off-policy evaluation and training** (inverse-propensity weighting, Section 13) so offline metrics estimate what *would* happen under a new policy, not just replay the old one.

### 3.3 Sequence, context, and the "right now" problem

Music intent is non-stationary within a single user. The same person wants lo-fi at 2pm and techno at midnight. This is why **sequential models** (Section 8) and **context features** (time, device, day-of-week, recent session) are not optional niceties — they are first-class signal. Frame the core retrieval task as **next-item / next-session prediction conditioned on the recent sequence and context**, not as static user→item matching.

### 3.4 Multiple objectives

"Did they play it" is not the only goal. Production rankers are **multi-task**: they jointly predict play probability, completion, save, skip, and downstream effects like *did this session keep the user listening longer*. The re-ranker then combines these heads into a single utility with tunable weights — which is how product priorities (engagement vs. discovery vs. creator equity) get encoded.

---

## 4. Data and feature foundations

Models are downstream of data. Get this layer right and everything else gets easier.

### 4.1 The core data model

You need three entity types and one event stream.

- **Users**: stable ID, registration country, subscription tier, long-term taste profile (aggregated), declared preferences.
- **Items (tracks)**: track ID, artist/album IDs, release date, language/market, editorial tags/genres, and — critically — **audio content** (or precomputed audio embeddings; Section 6). Items also carry the "12 sonic attributes" style features (danceability, energy, valence, tempo, acousticness, etc.) that Spotify exposes via its audio-features endpoint.
- **Context**: time, day-of-week, device, network, surface (Home, Search, Radio, playlist), and the immediately preceding tracks in the session.
- **Interaction events** (the fuel): `(user_id, item_id, timestamp, surface, action, completion_fraction, dwell, was_skip, position_shown)`. The `position_shown` field is gold — you need it for position-debiasing and off-policy evaluation.

### 4.2 Getting real data to practice on

You will not have Spotify's logs, so practice on these:

| Dataset | What it gives you | Good for |
|---|---|---|
| **Last.fm 1K / 360K** | Real user→track listening histories with timestamps | Sequential models, collaborative filtering |
| **Million Playlist Dataset (Spotify RecSys 2018)** | 1M real playlists, track co-occurrence | Playlist continuation, i2i, two-tower |
| **MovieLens-25M** | Clean implicit/explicit signal | Prototyping any architecture (used in Spotify's own Semantic-ID study) |
| **MusicBrainz / metadata dumps** | Catalog metadata | Content features, cold start |
| **Spotify Web API** | Audio features, related-artists, your own listening | Feature engineering, a personal demo |

A note on the **Spotify Web API**: it's excellent for *features and a personal prototype* (pull a user's top tracks, audio features, and build a content recommender), but it is rate-limited and not a substitute for a real interaction log. As of the mid-2020s some audio-analysis endpoints have been deprecated for new apps, so don't architect a production system around them — treat the API as a convenient feature source for learning.

### 4.3 Feature engineering that consistently pays off

- **Recency-weighted user taste vectors**: exponentially decay older interactions so the profile tracks current taste.
- **Sequence features**: last *N* item IDs (and their Semantic IDs), with positional/time-gap encoding.
- **Cross features for the ranker**: user-genre affinity × item-genre, user-artist play count, "is this the user's saved artist," etc. Early crossing is *allowed* in the ranker (Section 9) and is where a lot of precision comes from.
- **Context features**: hour-of-day bucket, is-weekend, device, surface.
- **Item popularity priors**, but **log-compressed and time-windowed** so you don't drown the catalog in head hits.

### 4.4 The golden rule: train/serve consistency

The single most common production bug class is **training/serving skew** — a feature computed one way in the training pipeline and another way at serving. Use a **feature store** (or at minimum a single shared transformation library) so the exact same code computes features offline and online. Log features *as served* whenever possible.

---

## 5. Representations I — collaborative and graph embeddings

Everything in the modern stack runs on **embeddings**: dense vectors where geometric closeness means "behaves similarly / is liked by similar people." There are three families of signal — collaborative, graph, and content — and the best systems fuse all three. This section covers the first two; Section 6 covers content/audio.

### 5.1 Matrix factorization with implicit feedback (the bedrock)

The oldest idea still worth knowing because it underpins the intuition for two-tower retrieval. Represent each user *u* and item *i* as vectors; predict preference as their dot product. For **implicit** data, train with confidence-weighted least squares (implicit ALS / WRMF) or **Bayesian Personalized Ranking (BPR)**, which optimizes a *ranking* objective: a user should score an item they interacted with above one they didn't.

```python
import torch, torch.nn as nn

class BPRMatrixFactorization(nn.Module):
    """Classic pairwise-ranking MF. A great, fast baseline — always build this first."""
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(-1) + self.item_bias(i).squeeze(-1)

    def bpr_loss(self, u, pos_i, neg_i):
        # maximize score(pos) - score(neg): the user prefers what they engaged with
        x_uij = self.score(u, pos_i) - self.score(u, neg_i)
        return -torch.log(torch.sigmoid(x_uij)).mean()
```

**Why it still matters:** BPR/ALS gives you (a) a strong baseline to beat — if your fancy transformer can't beat well-tuned MF, something is wrong; and (b) item embeddings you can directly index for i2i "more like this" retrieval.

### 5.2 Graph neural networks: model the interaction graph directly

User–item interactions *are* a bipartite graph. GNNs propagate information along edges so a user's embedding is informed by the items they touched, the *other* users who touched those items, and so on — capturing **high-order collaborative signal** that plain MF misses.

- **PinSage** (Pinterest) scaled GraphSAGE to web-scale item–item graphs using random-walk neighbor sampling — the proof that GNN recommenders work in production at billions of nodes.
- **LightGCN** is the one to actually implement. Its insight: for collaborative filtering you should **strip GCNs down to their essential operation — neighborhood aggregation — and drop the feature transforms and nonlinearities**, which only hurt. The final embedding is a weighted sum of embeddings at each propagation depth.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class LightGCN(nn.Module):
    """Simplified GCN for collaborative filtering. norm_adj is the symmetrically
    normalized (D^-1/2 A D^-1/2) sparse adjacency of the user-item bipartite graph."""
    def __init__(self, n_users, n_items, dim=64, n_layers=3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, norm_adj):
        x = self.emb.weight
        out = [x]
        for _ in range(self.n_layers):          # pure neighborhood aggregation, no weights/nonlinearity
            x = torch.sparse.mm(norm_adj, x)
            out.append(x)
        x_final = torch.stack(out, dim=1).mean(dim=1)   # layer-combination = average across depths
        return x_final[:self.n_users], x_final[self.n_users:]

    def bpr_loss(self, norm_adj, u, pos_i, neg_i):
        ue, ie = self.propagate(norm_adj)
        us, ps, ns = ue[u], ie[pos_i], ie[neg_i]
        pos = (us * ps).sum(-1); neg = (us * ns).sum(-1)
        return -F.logsigmoid(pos - neg).mean()
```

**When to reach for GNNs:** when you have rich co-interaction structure (playlists, co-listening) and want strong i2i and warm-user retrieval. They can be heavier to serve than two-tower; many shops precompute GNN embeddings offline and index them.

### 5.3 What you get out of this layer

Three sets of vectors you'll reuse everywhere: user embeddings, item embeddings, and (from playlists/co-listening) item–item similarity. These feed the two-tower retriever's initialization, the ranker's features, and — once quantized — the **Semantic IDs** of Section 11.

---

## 6. Representations II — audio and multimodal content embeddings

Collaborative signal is powerful but has one fatal gap: **it knows nothing about a track until people interact with it.** Content embeddings — learned from the audio itself and from text metadata — are how you recommend brand-new and long-tail items (the cold-start problem, Section 12) and how Spotify's content-based track representation works (it combines artist-sourced metadata, audio analysis, and, more recently, LLM-derived text understanding).

### 6.1 The three content signals

1. **Sonic attributes** — the interpretable, low-dimensional features (danceability, energy, valence, tempo, acousticness, instrumentalness, etc.). Cheap, useful for calibration and explanations, but too coarse to carry recommendation on their own.
2. **Learned audio embeddings** — dense vectors from a neural net that ingests the raw audio (as a log-mel spectrogram). This is where the state of the art lives.
3. **Text/semantic embeddings** — titles, descriptions, lyrics, editorial copy, transcripts (for podcasts), encoded by a text/LLM encoder. Spotify explicitly fuses "textual signals (what an item *is*)" with "behavioral signals (how listeners engage with it)."

### 6.2 State-of-the-art audio encoders

The historical baseline (still a great teaching example) is the **mel-spectrogram CNN**: turn audio into a 2D time–frequency image and run a convnet, predicting tags or collaborative targets. Sander Dieleman's classic Spotify-era work did exactly this — regress a CNN on audio to predict the *collaborative-filtering* latent vectors of tracks, so unseen songs get a usable embedding.

The 2024–2026 frontier moved to **self-supervised and contrastive** pretraining:

- **CLAP (Contrastive Language–Audio Pretraining)** — the audio analogue of CLIP. Two encoders (an audio encoder, typically HTS-AT, a Swin-Transformer over log-mel spectrograms; and a text encoder) are trained jointly with a contrastive loss so that an audio clip and its natural-language description land near each other. This gives you a *text-queryable* audio space ("upbeat melancholic synthwave") and embeddings that transfer well to recommendation.
- **MERT** — a self-supervised music understanding model (masked-prediction over audio, akin to BERT/HuBERT for music) that learns rich musical representations from unlabeled audio.

Recent research (2024–2026) specifically studies **adopting these pretrained MIR embeddings for recommender systems**, and finds contrastively pretrained neural audio embeddings (CLAP-style) are a promising drop-in for music recommendation, especially inside graph-based frameworks and for cold start.

You generally **do not train these from scratch.** Use a pretrained CLAP/MERT checkpoint as a frozen (or lightly fine-tuned) feature extractor:

```python
# Sketch: extract a CLAP audio embedding per track, then use it as a content feature.
# pip install transformers torchaudio
import torch, torchaudio
from transformers import ClapModel, ClapProcessor

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

@torch.no_grad()
def clap_audio_embedding(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != 48000:
        wav = torchaudio.functional.resample(wav, sr, 48000)
    inputs = proc(audios=wav.mean(0).numpy(), sampling_rate=48000, return_tensors="pt")
    emb = model.get_audio_features(**inputs)        # [1, D]
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

# Because CLAP is *language-aligned*, you also get zero-shot text query for free:
@torch.no_grad()
def clap_text_embedding(text):
    inputs = proc(text=[text], return_tensors="pt", padding=True)
    emb = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)
```

### 6.3 Fusing collaborative + content (the part that actually matters)

The winning pattern is a **hybrid** representation. A robust, production-friendly recipe:

1. Compute a collaborative item embedding `e_cf` (Section 5) for warm items.
2. Compute a content embedding `e_content` (CLAP/MERT + text) for *all* items.
3. Train a small projection head that maps `e_content → e_cf` space (so cold items can be placed in the collaborative geometry), and fuse: `e_item = MLP([e_cf ⊕ e_content])`, with `e_cf` replaced by the content-predicted vector when the item is cold.

```python
class HybridItemEncoder(nn.Module):
    """Fuse collaborative + content; gracefully falls back to content-only for cold items."""
    def __init__(self, cf_dim, content_dim, out_dim=128):
        super().__init__()
        self.content2cf = nn.Sequential(           # predict CF vector from content (cold-start bridge)
            nn.Linear(content_dim, cf_dim), nn.ReLU(), nn.Linear(cf_dim, cf_dim))
        self.fuse = nn.Sequential(
            nn.Linear(cf_dim + content_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))

    def forward(self, e_cf, e_content, is_cold):
        predicted_cf = self.content2cf(e_content)
        e_cf_eff = torch.where(is_cold.unsqueeze(-1).bool(), predicted_cf, e_cf)
        return self.fuse(torch.cat([e_cf_eff, e_content], dim=-1))
```

This single idea — **predict the collaborative embedding from content so new songs are usable on day one** — is the backbone of practical music cold start, and it is conceptually what Semantic IDs do at the token level (Section 11).

---

## 7. Stage 1 — candidate retrieval with a two-tower model

This is the workhorse retriever. The job: from ~50M items, return a few thousand plausible candidates **in a few milliseconds**, with high *recall* (don't miss the good stuff) — precision is the ranker's job.

### 7.1 The architecture and why it's shaped this way

Two separate networks: a **user/query tower** encodes the user + context into a vector; an **item tower** encodes a track into a vector in the *same* space. Relevance = dot product (or cosine). The non-negotiable constraint: **the towers don't interact until the final dot product** ("late interaction"). That restriction is what lets you **precompute every item vector offline, build an ANN index once, and serve retrieval as: one user forward pass → ANN top-K.** This is the candidate-generation model from the YouTube deep-recommendations paper, generalized.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), out_dim=64):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h)]; d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)     # L2-normalize → dot product == cosine

class TwoTower(nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, out_dim=64):
        super().__init__()
        self.user_tower = Tower(user_feat_dim, out_dim=out_dim)
        self.item_tower = Tower(item_feat_dim, out_dim=out_dim)

    def forward(self, user_feats, item_feats):
        return self.user_tower(user_feats), self.item_tower(item_feats)
```

In a real system each tower is richer: the **user tower** ingests the recent-item sequence (often via a small transformer — this is where Section 8 plugs in), aggregated taste vectors, and context; the **item tower** ingests the hybrid content+collaborative embedding from Section 6 plus metadata.

### 7.2 The training objective: in-batch negatives + sampled softmax

You don't have explicit negatives, and scoring all 50M items per step is impossible. The standard trick: **in-batch negatives.** Within a training batch of *B* matched (user, positive-item) pairs, treat the *other* B−1 items in the batch as negatives for each user. One matrix multiply gives you a B×B score matrix; the diagonal is the positives.

```python
def in_batch_softmax_loss(user_emb, item_emb, temperature=0.05, log_q=None):
    """user_emb, item_emb: [B, d], row i is a matched positive pair.
    log_q: optional log of sampling prob per in-batch item for popularity correction."""
    logits = user_emb @ item_emb.t() / temperature          # [B, B]
    if log_q is not None:                                    # logQ correction: subtract popularity bias
        logits = logits - log_q.unsqueeze(0)
    labels = torch.arange(user_emb.size(0), device=user_emb.device)
    return F.cross_entropy(logits, labels)                   # diagonal = positives
```

Two details that separate a toy from a real retriever:

- **Popularity (logQ) correction.** In-batch negatives oversample popular items (they appear often), biasing the model to over-recommend hits. Subtract `log(sampling_prob)` from the logits — the **sampled-softmax / logQ correction** from the YouTube/Google two-tower literature. Without it your retriever becomes a popularity machine.
- **Temperature** sharpens the softmax; tune it (≈0.05–0.1). It materially affects recall.
- **Mixed negatives.** In-batch negatives are "easy." Add a stream of **hard negatives** (e.g., items the user was shown but skipped, or ANN-mined near-misses) to sharpen decision boundaries.

### 7.3 Serving: ANN with FAISS

Precompute all item vectors, index them, query with the user vector.

```python
import faiss, numpy as np

# Build index once (offline). item_vecs: [N, d] float32, L2-normalized.
d = item_vecs.shape[1]
index = faiss.IndexHNSWFlat(d, 32)          # HNSW: fast, high-recall graph index
index.metric_type = faiss.METRIC_INNER_PRODUCT
index.add(item_vecs)

# At serving (online), for a freshly computed user vector:
def retrieve(user_vec, k=1000):
    scores, ids = index.search(user_vec.reshape(1, -1).astype('float32'), k)
    return ids[0], scores[0]
```

Use **IVF-PQ** instead of flat HNSW when the catalog gets huge and memory matters (product quantization compresses vectors at a small recall cost). Refresh item embeddings on a schedule; new items get embedded and added incrementally.

### 7.4 Many retrievers, not one

Production retrieval is an **ensemble of sources** whose candidates are unioned before ranking: the two-tower ANN, a sequential next-item model (Section 8), item–item co-listening, content/audio similarity (for cold start and discovery), editorial/fresh pools, and popularity fallbacks. Each covers a different failure mode. Spotify's research on **RADAR** (Recall Augmentation through Deferred Asynchronous Retrieval) is an example of adding asynchronous retrieval passes to widen recall beyond a single synchronous retriever.

---

## 8. Stage 2 — sequential modeling (SASRec / BERT4Rec)

Static user→item matching ignores order. **Sequential recommenders** predict the next item from the *ordered* history, capturing short-term intent and session dynamics — exactly what music needs. These models serve as a retriever (next-item generation), as the user tower's sequence encoder, and as the conceptual bridge to generative recommendation (Section 11).

### 8.1 SASRec — self-attentive, left-to-right

SASRec applies a **causal (left-to-right) Transformer** over the user's item sequence: at each position it predicts the next item. Causal masking means position *t* only attends to positions ≤ *t*. It decisively beat the older GRU/RNN sequential models (GRU4Rec) — evidence that self-attention is the better tool for this task.

```python
import torch, torch.nn as nn

class SASRec(nn.Module):
    """Self-Attentive Sequential Recommendation (causal transformer, next-item objective)."""
    def __init__(self, n_items, max_len=200, dim=64, n_heads=2, n_blocks=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)   # 0 = pad
        self.pos_emb  = nn.Embedding(max_len, dim)
        enc = nn.TransformerEncoderLayer(dim, n_heads, dim*4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_blocks)
        self.max_len, self.dropout = max_len, nn.Dropout(dropout)

    def forward(self, seq):                          # seq: [B, L] item ids (0-padded on the left)
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, device=seq.device), diagonal=1).bool()  # no peeking ahead
        pad_mask = (seq == 0)
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad_mask)
        return h                                       # [B, L, dim]; h[:, -1] = next-item user state

    def score_all(self, seq):
        h_last = self.forward(seq)[:, -1]              # [B, dim]
        return h_last @ self.item_emb.weight.t()       # [B, n_items+1] logits over catalog
```

Train it with sampled-softmax / cross-entropy over the next item at each position (shifted sequence as labels). At serving, take the last hidden state as the **user-state vector** — and note it slots directly into the two-tower user tower, or you ANN-search the item embedding table with it.

### 8.2 BERT4Rec — bidirectional, masked (Cloze)

BERT4Rec uses a **bidirectional** Transformer trained with a masked-item (Cloze) objective: randomly mask items in the sequence and predict them from both sides. Bidirectional context can model "this item sits between these others" relationships that a left-to-right model can't see during training. It reported strong gains over SASRec and GRU4Rec on standard benchmarks — though later replication work ("Turning Dross into Gold Loss") showed that **SASRec, properly trained with a full softmax / many negatives rather than a single sampled negative, is competitive with or better than BERT4Rec.** The practical lesson: *the loss and negative sampling matter as much as the architecture.*

### 8.3 Which to use

Start with **SASRec + full/sampled softmax over the catalog**; it's simpler to serve (causal models extend naturally to streaming/autoregression) and, tuned well, is a top baseline. Use BERT4Rec when you have a genuinely bidirectional use case (e.g., playlist infilling). Both are stepping stones to **HSTU** and generative recommenders, which are essentially these sequence models scaled up and pointed at Semantic-ID vocabularies (Section 11).

---

## 9. Stage 3 — ranking (multi-task DLRM / DCN)

Retrieval handed you ~1–10k candidates. Now spend real compute to order them precisely. The ranker is **allowed early feature crossing** (it only runs on a small set), so this is where rich user×item interactions and many objectives come in.

### 9.1 What's different from retrieval

- **Inputs**: full cross features (user-artist play count, user-genre × item-genre, "saved artist?", session context, the retrieval scores themselves as features).
- **Architecture**: a deep network that *explicitly models feature interactions*. The lineage: Wide & Deep → DeepFM → **DCN / DCNv2 (Deep & Cross Network)** → DLRM. DCN's "cross network" learns bounded-degree feature crossings automatically, which is ideal for the many sparse categorical features in recommendation.
- **Objective**: **multi-task.** Predict several behaviors at once with shared lower layers and per-task heads, then combine.

### 9.2 A multi-task ranking model

```python
import torch, torch.nn as nn

class CrossNetwork(nn.Module):
    """DCN cross layers: explicit, efficient bounded-degree feature interactions."""
    def __init__(self, in_dim, n_layers=3):
        super().__init__()
        self.w = nn.ParameterList([nn.Parameter(torch.randn(in_dim, 1) * 0.01) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(n_layers)])

    def forward(self, x0):
        x = x0
        for w, b in zip(self.w, self.b):
            x = x0 * (x @ w) + b + x          # x_{l+1} = x0 ⊙ (x_l·w) + b + x_l
        return x

class MultiTaskRanker(nn.Module):
    """Shared bottom + DCN + per-objective heads. Outputs calibrated probabilities per task."""
    def __init__(self, in_dim, tasks=("play", "complete", "save", "skip"), hidden=(512, 256)):
        super().__init__()
        self.cross = CrossNetwork(in_dim, n_layers=3)
        deep, d = [], in_dim
        for h in hidden:
            deep += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]; d = h
        self.deep = nn.Sequential(*deep)
        self.heads = nn.ModuleDict({t: nn.Linear(in_dim + hidden[-1], 1) for t in tasks})
        self.tasks = tasks

    def forward(self, x):
        z = torch.cat([self.cross(x), self.deep(x)], dim=-1)
        return {t: self.heads[t](z).squeeze(-1) for t in self.tasks}   # logits per task

def multitask_loss(logits, labels, weights):
    # labels[t], weights[t] per task; skip is a "negative" objective (penalize)
    return sum(weights[t] * nn.functional.binary_cross_entropy_with_logits(logits[t], labels[t])
               for t in logits)
```

### 9.3 Combining heads into one score

The re-ranker (or a final scoring step) blends task probabilities into a single utility:

```
utility = w_play·P(play) + w_complete·P(complete) + w_save·P(save)
          + w_listen·E[listen_time] − w_skip·P(skip)
```

These weights are **product levers**, often tuned via online A/B tests or Pareto/multi-objective optimization. They're how you dial "engagement vs. discovery." Watch for **task imbalance** (skips ≫ saves) and use techniques like uncertainty weighting or gradient balancing if one task dominates.

### 9.4 Calibration

Downstream policy (Section 10) needs **calibrated** probabilities, not just correct ordering — a "0.8 play probability" must mean 80%. Check reliability diagrams; apply Platt scaling / isotonic regression if heads are miscalibrated. Position bias correction (model the probability an item is examined at a given slot, e.g. with a position feature you zero out at serving) belongs here too.

---

## 10. Stage 4 — re-ranking, diversity, calibration, and bandits

A perfectly ranked list by predicted utility is often a *bad* slate: ten near-identical songs, all exploit and no explore, no room for new artists. The final stage turns scores into a good *experience*.

### 10.1 Diversity: MMR and DPPs

**Maximal Marginal Relevance (MMR)** greedily builds a list trading relevance against redundancy:

```python
import numpy as np

def mmr_rerank(cand_ids, relevance, item_vecs, k=20, lam=0.7):
    """lam→1 favors pure relevance; lam→0 favors diversity."""
    selected, pool = [], list(range(len(cand_ids)))
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: relevance[i])
        else:
            sims = {i: max(item_vecs[i] @ item_vecs[j] for j in selected) for i in pool}
            best = max(pool, key=lambda i: lam * relevance[i] - (1 - lam) * sims[i])
        selected.append(best); pool.remove(best)
    return [cand_ids[i] for i in selected]
```

**Determinantal Point Processes (DPPs)** are the more principled tool: they model a slate's quality *and* diversity jointly via a kernel, sampling sets that are both relevant and internally dissimilar.

### 10.2 Calibration to the user's taste mix

Beyond intra-list diversity, **calibrated recommendation** ensures the *proportions* match the user's actual taste: if someone listens 70% hip-hop / 30% jazz, the slate shouldn't be 100% hip-hop just because those scores are marginally higher. Spotify Research describes calibrated recommendations driven by **contextual bandits on the Home page** for exactly this. Penalize KL divergence between the genre distribution of the slate and the user's historical distribution.

### 10.3 Exploration: bandits (this is BaRT)

The selection-bias loop from Section 3 is fatal if you only ever show top-utility items. **Multi-armed / contextual bandits** inject principled exploration. Spotify's Home engine **BaRT** uses a bandit framework (an ε-greedy policy) to decide which shelves and cards to show, explicitly balancing exploitation (known favorites) and exploration (discovery), and learning from the resulting feedback.

```python
import numpy as np

def epsilon_greedy(scored_items, epsilon=0.1):
    """With prob epsilon explore (uniform), else exploit the best. The simplest BaRT-style policy."""
    if np.random.rand() < epsilon:
        return np.random.choice(len(scored_items))
    return int(np.argmax(scored_items))
```

In practice you'd use **LinUCB / Thompson sampling (contextual bandits)** so exploration is *informed* by features rather than uniform, and you log propensities (the probability you showed each item) — which you need for the off-policy evaluation in Section 13.

### 10.4 Business and well-being constraints

The final policy also enforces: artist/label diversity and creator-equity rules, freshness and fatigue (don't replay the same track to exhaustion), licensing/market availability, and explicit-content filters. This is the layer where non-ML product policy lives, and keeping it *separate* from the learned scores keeps the system debuggable.

---

## 11. The frontier — generative recommendation and Semantic IDs

This is the part you specifically asked about: **the most state-of-the-art way to use AI here.** The classic funnel (Sections 7–10) is mature and excellent, but it has structural limits — it leans on memorized per-item ID embeddings (bad for cold start and the long tail), it's a pipeline of separately-trained models that can't share signal end-to-end, and crucially **classic DLRMs don't reliably get better as you scale them.** Generative recommendation attacks all three.

### 11.1 The core idea: items as token sequences

Give every item a **Semantic ID**: a short sequence of discrete codes derived from its embedding, such that *semantically similar items share prefix tokens*. A track might become `(12, 7, 41, 3)`. Then represent a user as the concatenated Semantic IDs of their history, and train a Transformer to **autoregressively generate the next item's Semantic ID** — recommendation as next-token prediction, exactly like an LLM.

Why this is powerful:
- **Generalization & cold start.** A new track gets a Semantic ID from its content embedding the moment it's ingested — no waiting for interactions, and it automatically sits near similar items because they share tokens. Spotify highlights this "similar items share tokens → better generalization" property directly.
- **Compactness & efficiency.** Discrete tokens cut memory and bandwidth versus giant ID-embedding tables.
- **Unification.** The same generative model can serve **search and recommendation** at once. Spotify found multi-task (search + rec) training gave an *additional 22% improvement* over single-task — evidence the model learns a shared user/item structure.
- **It scales.** See HSTU below.

### 11.2 How Semantic IDs are built (quantization)

You take a continuous item embedding (collaborative, content, or both — Section 6) and **quantize** it into discrete codes. The dominant methods:

- **RQ-VAE (Residual-Quantized VAE)** — the method in Google's **TIGER**. An encoder maps the item to a latent; you quantize it against a codebook, take the *residual*, quantize that against a second codebook, and so on. The result is a hierarchy of codes: coarse-to-fine, which mirrors how an LLM generates tokens. (Related quantizers: RVQ, residual k-means / **RQ-KMeans**, and Spotify's internally-developed residual **Lookup-Free Quantization (LFQ)**.)
- The coarse code picks a broad region of embedding space ("general topic"); residual codes refine it step by step. This hierarchical refinement *is* a short token sequence — a natural fit for autoregressive modeling.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RQVAE(nn.Module):
    """Residual-Quantized VAE → Semantic IDs. Encode item embedding into `n_codebooks` tokens."""
    def __init__(self, in_dim, latent_dim=32, n_codebooks=3, codebook_size=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim))
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, latent_dim)) for _ in range(n_codebooks)])

    def quantize(self, z):
        codes, residual, zq = [], z, 0
        for cb in self.codebooks:                       # quantize the residual at each level
            dist = torch.cdist(residual, cb)            # [B, codebook_size]
            idx = dist.argmin(dim=-1)                    # nearest codeword = this level's token
            q = cb[idx]
            zq = zq + q
            residual = residual - q                      # pass the residual to the next codebook
            codes.append(idx)
        return zq, torch.stack(codes, dim=-1)            # zq: reconstruction, codes: [B, n_codebooks]

    def forward(self, x):
        z = self.encoder(x)
        zq, codes = self.quantize(z)
        x_hat = self.decoder(z + (zq - z).detach())      # straight-through estimator
        recon = F.mse_loss(x_hat, x)
        commit = F.mse_loss(z, zq.detach())              # commitment loss keeps encoder near codebook
        return x_hat, codes, recon + 0.25 * commit
```

A practical wrinkle TIGER handles: collisions (two items mapping to the same code tuple) are resolved by appending a disambiguating token.

### 11.3 TIGER — generative retrieval

**TIGER (Transformer Index for GEnerative Recommenders)** trains a seq2seq Transformer where the input is the user's history as Semantic-ID tokens and the output is the next item's Semantic ID, generated autoregressively (with beam search to produce multiple candidates). It *replaces the ANN index entirely* — the model "generates" the IDs to retrieve. It set state-of-the-art results and gave strong cold-start behavior precisely because of the shared-token structure.

```python
# Conceptual training shape for a TIGER-style generative retriever.
# Vocabulary = special tokens + (n_codebooks × codebook_size) semantic-ID tokens.
# Input  : [<bos>, sid(i1)_0, sid(i1)_1, sid(i1)_2,  sid(i2)_0, ... ]  (user history)
# Target : [ sid(i_next)_0, sid(i_next)_1, sid(i_next)_2, <eos> ]
#
# Use any encoder-decoder (e.g., a small T5). At inference, beam-search the decoder to emit
# K valid Semantic-ID sequences → map each back to a catalog item (via a trie of valid IDs
# so you never generate a non-existent item). That beam set IS your candidate list.
```

The serving subtlety: you must **constrain decoding to valid IDs** (a prefix trie / constrained beam search), so the model can only emit Semantic IDs that correspond to real catalog items.

### 11.4 HSTU and the recommendation scaling law (Meta)

Meta's **"Actions Speak Louder than Words"** (ICML 2024) reformulated ranking *and* retrieval as a single generative sequence problem over user actions, with a new attention architecture, **HSTU (Hierarchical Sequential Transduction Unit)**, designed for high-cardinality, non-stationary streaming data. Headline results you should know:

- HSTU outperforms baselines by up to **65.8% NDCG** and is **5.3×–15.2× faster** than FlashAttention-2 Transformers on long (8192) sequences.
- Most importantly: **model quality scales as a power law of training compute across three orders of magnitude, up to GPT-3/LLaMA-2 scale** — the *first demonstration that scaling laws apply to recommendation.* This is the crux of why people call generative recommendation a potential "ChatGPT moment" for RecSys: you can now buy quality with compute, which was never reliably true for ID-embedding DLRMs.
- Deployed at 1.5T parameters, it improved online A/B metrics by 12.4% on a platform with billions of users. (Meta open-sourced the `generative-recommenders` code.)

### 11.5 LLM-as-recommender: "teaching an LLM to speak Spotify"

The newest line (2025–2026) folds the actual **LLM** into the recommender. Spotify domain-adapts an **open-weight LLM** so it can read and write **Semantic IDs as new vocabulary tokens**, letting it reason over the catalog the way it reasons over words. The recipe, faithfully:

1. **Build Semantic IDs** for tens of millions of entities by fusing *textual* signals (titles, descriptions, transcripts → "what an item is") and *behavioral* signals (co-listening, transitions → "how listeners engage"), then quantizing with **residual LFQ**. Type-specific quantizers (music vs. podcast) share a backbone so everything is comparable.
2. **Expand the LLM's vocabulary** with the new Semantic-ID tokens; enlarge the embedding matrix. **Random initialization** of the new tokens worked best. Align them to language with a **partial-freeze** scheme — freeze the core LLM, train only the new token embeddings on mixed text + Semantic-ID sequences — so the model learns that "melancholic piano" maps to specific catalog entities.
3. **Domain fine-tune** on personalization + reasoning tasks (episode recommendation, search, playlist generation, "explain why"), interleaving natural language and Semantic IDs, mixing in a little text-only instruction data to prevent **catastrophic forgetting**, plus **synthetic** query/ID data.

Results worth internalizing: a **1B-parameter** model matched or beat production baselines, with up to **1.96× improvement on episode recommendation**; multi-task (search + rec) added **+22%**; cleaning the input text improved accuracy up to **5.4%** (better text → better embeddings → cleaner quantization → more distinct Semantic IDs). Scaling 0.5B → 8B gave up to **+16%** on search, but they deliberately ship **smaller** models to meet real-time latency. Serving uses **vLLM** with **beam search** over Semantic IDs (constrained so outputs are valid catalog items), and a **Redis** key-value store to translate Spotify URIs ↔ Semantic IDs on the fly. A standout capability: the model can **explain** a recommendation in natural language grounded in the listener's history — something the classic funnel simply cannot do.

Related: Google's follow-up showed Semantic IDs also improve **ranking** generalization, not just retrieval; and Meta/industry **PLUM** work adapts pretrained LLMs for industrial-scale generative recommendation. The "Practitioner's Handbook" on Semantic IDs (arXiv 2507.22224) is the best single implementation reference.

### 11.6 How to actually adopt this (without a research lab)

A realistic progression for a small team:
1. Build the classic funnel first (Sections 5–10). You need its data, labels, and evaluation harness regardless.
2. Train an **RQ-VAE/RQ-KMeans** to turn your hybrid item embeddings into Semantic IDs.
3. Train a **TIGER-style seq2seq** retriever on next-item Semantic-ID generation; add it as **one more retriever** in your union (Section 7.4) and measure recall lift, *especially on cold/long-tail items*.
4. Feed Semantic IDs as **features into your ranker** (Section 9) — a cheap, reliable win.
5. Only if you have the budget: domain-adapt an open-weight LLM (1B class) with Semantic-ID vocabulary for unified search/rec + explanations.

---

## 12. Cold start

Three flavors, three remedies:

- **New item (new track).** Embed it from **content** the moment it lands (CLAP/MERT + text), project into collaborative space (Section 6.3), and/or assign a **Semantic ID** so it's instantly retrievable near similar items. This is the single biggest reason content embeddings and Semantic IDs matter.
- **New user.** Lean on **context + popularity + onboarding** (the "pick 3 artists you like" flow seeds a taste vector), then let **exploration** (Section 10.3) gather signal fast. Session-based models help because they work from in-session behavior, not long history.
- **New context / surface.** Multi-task and contextual features let a warm model generalize to a new placement; bandits handle the residual uncertainty.

A clean mental model: **collaborative filtering is interpolation among known behavior; content/Semantic-ID methods are how you extrapolate to the unknown.** A production system needs both.

---

## 13. Evaluation: offline, counterfactual, and online

Evaluation is where most recommender projects quietly fail. Use three tiers.

### 13.1 Offline metrics (fast, biased)

Compute on a **temporally held-out** split (train on the past, test on the future — never random split a sequential problem):

- **Ranking quality:** Recall@K and NDCG@K (graded relevance), MAP, MRR. NDCG@K is the workhorse.
- **Beyond accuracy:** **coverage** (fraction of catalog ever recommended), **diversity** (intra-list dissimilarity), **novelty / serendipity**, **popularity bias** (are you just serving hits?), and **calibration** (does the genre mix match the user's?).

```python
import numpy as np

def ndcg_at_k(ranked_ids, relevant_set, k=10):
    dcg = sum((1.0 / np.log2(rank + 2))
              for rank, iid in enumerate(ranked_ids[:k]) if iid in relevant_set)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant_set), k)))
    return dcg / idcg if idcg > 0 else 0.0

def catalog_coverage(all_recommended_ids, catalog_size):
    return len(set(all_recommended_ids)) / catalog_size
```

The trap: offline accuracy only measures your ability to **replay the logged policy**. A model that perfectly predicts past plays may tank in production because it never explores. Always pair accuracy with coverage/diversity.

### 13.2 Counterfactual / off-policy evaluation (the bridge)

Before an expensive A/B test, estimate how a *new* policy would have performed using *old* logs, correcting for the fact that the old policy chose what you observed. **Inverse Propensity Scoring (IPS)** reweights logged rewards by 1/(probability the logging policy showed that item) — which is exactly why you **logged propensities** in Section 10.3. Variants: capped IPS, doubly-robust estimators. This is also why Spotify frames recommendations "as treatments" (the *T* in BaRT) — it's the causal-inference lens on the whole problem.

### 13.3 Online A/B testing (the truth)

The only ground truth. Randomize users into control/treatment and measure the metrics you actually care about over time: not just clicks/plays, but **retention, day-2/day-30 return, session length, save rate, and discovery** (are users finding new artists?). Beware short-term/long-term tension: a model that maximizes immediate plays by serving only hits can *erode* long-term retention. Guardrail metrics catch this.

---

## 14. Serving and system architecture

A reference layout that maps to everything above:

```
            OFFLINE (batch / streaming training)              ONLINE (per-request, ~tens of ms)
  ┌───────────────────────────────────────────┐     ┌────────────────────────────────────────┐
  │ • Event ETL → labels (Sec 3,4)             │     │ Request: (user, context, surface)        │
  │ • Train embeddings (Sec 5,6)               │     │   │                                      │
  │ • Train two-tower / SASRec / TIGER (7,8,11)│     │   ├─► Feature fetch (feature store)       │
  │ • Train ranker (Sec 9)                     │     │   ├─► User-tower fwd pass → user vector   │
  │ • Build Semantic IDs (Sec 11)              │ ──► │   ├─► RETRIEVE: ANN (FAISS) + i2i +       │
  │ • Compute & publish item embeddings        │     │   │     generative + popularity (union)   │
  │ • Build ANN index, Semantic-ID KV store    │     │   ├─► RANK: multi-task model on ~1–10k    │
  └───────────────────────────────────────────┘     │   ├─► RE-RANK: diversity/calibration/     │
            │  push artifacts to online stores       │   │     bandit policy (Sec 10)            │
            ▼                                         │   └─► slate + impression/propensity log   │
   Feature store · Vector index · KV store · Model servers       │ (feeds back to OFFLINE) ◄────────┘
```

Practical notes:
- **Latency budget** (typically <100ms end-to-end) is the master constraint. It's why retrieval uses ANN, why rankers run on thousands not millions, and why Spotify ships a *small* LLM despite larger ones scoring higher.
- **Freshness:** item embeddings and indexes refresh on a schedule; new items get embedded/Semantic-ID'd continuously. User sequence features update in near-real-time within a session.
- **Logging is part of the system, not an afterthought:** log features-as-served, positions, and propensities — your future training and off-policy evaluation depend on it.
- **Serving generative models:** vLLM-style high-throughput LLM serving + constrained beam search over Semantic IDs + a Redis URI↔Semantic-ID map is the documented Spotify pattern.

---

## 15. A concrete build roadmap

A phased plan that always leaves you with a working system.

**Phase 0 — Data & evaluation harness (week 1–2).** Build the event→label pipeline (Sec 3–4) and the offline eval (NDCG/Recall/coverage, temporal split, Sec 13). *Don't model anything until you can measure it.*

**Phase 1 — Baselines (week 2–3).** Popularity, then **BPR/implicit-ALS matrix factorization** (Sec 5.1). This is your bar. Ship it as the i2i "more like this" feature.

**Phase 2 — Two-tower retrieval + FAISS (week 3–5).** In-batch softmax with logQ correction (Sec 7). Add hard negatives. Now you have scalable candidate generation.

**Phase 3 — Sequential model (week 5–7).** SASRec (Sec 8) as a second retriever and as the user-tower sequence encoder. Expect a real lift from session awareness.

**Phase 4 — Multi-task ranker (week 7–9).** DCN + per-objective heads (Sec 9), then a re-ranking policy with diversity + ε-greedy exploration (Sec 10). You now have the full classic funnel.

**Phase 5 — Content & cold start (week 9–11).** CLAP/MERT audio + text embeddings, hybrid fusion (Sec 6), content-similarity retriever. Cold-start coverage jumps.

**Phase 6 — Generative frontier (week 11+).** RQ-VAE → Semantic IDs; a TIGER-style generative retriever added to the union; Semantic IDs as ranker features (Sec 11). Optional, budget-permitting: LLM domain-adaptation for unified search/rec + explanations.

At every phase: A/B against the previous phase, watch guardrails (retention, diversity), and keep exploration on so your logs stay healthy.

---

## 16. Reading list and primary sources

**Spotify (production & frontier):**
- Semantic IDs for Generative Search and Recommendation — Spotify Research (2025): https://research.atspotify.com/2025/9/semantic-ids-for-generative-search-and-recommendation
- Teaching LLMs to "Speak Spotify": How Semantic IDs Enable Personalization — Spotify Research (2025): https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable
- Calibrated Recommendations with Contextual Bandits on Spotify Homepage — Spotify Research (2025): https://research.atspotify.com/2025/9/calibrated-recommendations-with-contextual-bandits-on-spotify-homepage
- Inside Spotify's Recommendation System (2025 guide) — Music Tomorrow: https://www.music-tomorrow.com/blog/how-spotify-recommendation-system-works-complete-guide
- BaRT overview: https://dynamoi.com/learn/spotify-algorithm
- RADAR: Recall Augmentation through Deferred Asynchronous Retrieval: https://arxiv.org/pdf/2506.07261

**Multi-stage funnel & retrieval/ranking:**
- Covington et al., *Deep Neural Networks for YouTube Recommendations* (RecSys 2016): https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/
- Two-tower deep dive — Shaped: https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive
- Two-tower & negative sampling — Towards Data Science: https://towardsdatascience.com/two-tower-networks-and-negative-sampling-in-recommender-systems-fdc88411601b/
- Wang et al., *DCN / DCN-v2 (Deep & Cross Network)* — feature crossing for ranking.

**Embeddings (collaborative & graph):**
- Hu, Koren, Volinsky, *Collaborative Filtering for Implicit Feedback* (2008) — confidence-weighted ALS.
- Rendle et al., *BPR: Bayesian Personalized Ranking* (2009).
- He et al., *LightGCN* (SIGIR 2020): https://arxiv.org/abs/2002.02126
- Ying et al., *PinSage* (KDD 2018) — web-scale GNN recommender.

**Sequential models:**
- Kang & McAuley, *SASRec* (ICDM 2018).
- Sun et al., *BERT4Rec* (CIKM 2019): https://arxiv.org/pdf/1904.06690
- Petrov & Macdonald, *Turning Dross into Gold Loss: is BERT4Rec really better than SASRec?* (RecSys 2023): https://arxiv.org/pdf/2309.07602

**Audio / content representations:**
- Dieleman & Schrauwen, *Deep content-based music recommendation* (NeurIPS 2013) — CNN-predicts-CF-vectors, the cold-start classic.
- Elizalde et al., *CLAP: Learning Audio Concepts from Natural Language Supervision* (2022).
- Li et al., *MERT* — self-supervised music understanding.
- *Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems* (2026): https://arxiv.org/html/2604.23077v1
- *Towards Leveraging Contrastively Pretrained Neural Audio Embeddings for Recommender Tasks* (2024): https://arxiv.org/html/2409.09026v1

**Generative recommendation (the frontier):**
- Rajput et al., *TIGER: Recommender Systems with Generative Retrieval* (NeurIPS 2023).
- Singh et al., *Better Generalization with Semantic IDs: A Case Study in Ranking* (RecSys/NeurIPS 2024).
- Zhai et al., *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers (HSTU)* (ICML 2024): https://arxiv.org/abs/2402.17152 · code: https://github.com/meta-recsys/generative-recommenders
- *Generative Recommendation with Semantic IDs: A Practitioner's Handbook* (2025): https://arxiv.org/html/2507.22224v1
- He et al., *PLUM: Adapting Pre-trained LMs for Industrial-scale Generative Recommendations* (2025).
- "Is Generative Recommendation the ChatGPT Moment of RecSys?" — Yuan Meng: https://www.yuan-meng.com/posts/generative_recommendation/

---

### A final word on "state of the art"

There are two honest answers to "what's the most state-of-the-art method." The **production** answer is a well-tuned multi-stage funnel with strong embeddings, a two-tower retriever, a sequential model, a multi-task ranker, and a bandit-driven exploration/diversity policy — this is what reliably moves metrics today. The **frontier** answer is **generative recommendation with Semantic IDs** — TIGER-style generative retrieval, HSTU-style scaling, and LLMs domain-adapted to "speak" your catalog — which is where Spotify, Google, and Meta are actively investing because, uniquely, it keeps improving with scale and unifies search, recommendation, and explanation. Build the funnel to learn the discipline and ship value; layer in Semantic IDs and generative retrieval to reach the frontier. The code scaffolds above give you a runnable starting point for every stage.






