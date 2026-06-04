# Building a Batch Book Recommender (Goodreads-Style)
### A practitioner's guide to recommendation from historical preferences — explicit ratings, content & NLP, hybrids, and the LLM/RAG frontier, all served in batch

> **Who this is for.** ML engineers comfortable with Python, PyTorch, embeddings, and classic ML who want a faithful, end-to-end map of how to build a book recommender on Goodreads-style data — where preferences are *stable*, feedback is *explicit* (1–5 stars), and the system can be retrained and served in **batch** rather than as a constant stream.
>
> **The contrast with streaming (e.g. music).** A Spotify-style recommender lives or dies on real-time session signal and implicit feedback (plays/skips). A book recommender is the opposite regime: taste evolves over weeks, the catalog changes slowly, feedback is graded and sparse, and a **nightly retrain → precompute top-N → serve from a key-value store** architecture captures essentially all the signal with negligible staleness. That single fact reshapes every design choice below.
>
> **How to read it.** Sections 1–4 set the mental model, the explicit-vs-implicit distinction, the data, and problem framing. Sections 5–7 build the core: collaborative baselines, content/NLP representations, and hybrids (the cold-start backbone). Sections 8–9 cover batch two-tower retrieval and ranking/re-ranking. Section 10 is the frontier: LLM-based recommendation, RAG/conversational, and generative retrieval — with a hard focus on *grounding* so you never recommend a book that doesn't exist. Sections 11–13 are evaluation (the part most projects get wrong), batch serving, and a phased roadmap. Section 14 is the sourced reading list.

---

## Table of contents

1. The big picture: batch recommendation from historical preferences
2. Explicit vs implicit feedback — the distinction that defines this domain
3. The Goodreads data model and public datasets
4. Framing the problem (rating prediction vs ranking, MNAR, cold start)
5. Collaborative filtering baselines (kNN, biased MF, SVD++, ALS, BPR)
6. Content and NLP representations (TF-IDF → embeddings → topics/sentiment → knowledge graphs)
7. Hybrid models and cold start (LightFM, two-tower, stacking)
8. Batch retrieval with a two-tower model + ANN
9. Ranking, re-ranking, and beyond-accuracy objectives
10. The frontier: LLMs, RAG/conversational, and generative retrieval
11. Evaluation: the part most projects get wrong
12. Batch serving architecture
13. A concrete build roadmap
14. Reading list and primary sources

---

## 1. The big picture: batch recommendation from historical preferences

A book recommender answers a deceptively simple question: *given everything this reader has rated and shelved over the years, which books should we put in front of them next?* The defining property — and the reason this guide exists as a separate document from a streaming/music one — is that **almost nothing about the answer changes minute to minute.**

Three structural facts drive every decision:

1. **Preferences are stable and slow-moving.** A reader's taste drifts over weeks and months, not within a session. There is little value in updating recommendations the instant they finish a chapter. This is what makes **batch** (periodic retrain + precomputed recommendations) not just acceptable but *optimal* — you get near-100% of the signal with a fraction of the operational complexity of a streaming stack.
2. **Feedback is explicit and graded.** Goodreads gives you 1–5 star ratings. Unlike plays/skips, a 2★ is an unambiguous "read it, disliked it" — a real negative you can model directly. You can predict an actual rating, report RMSE, and distinguish degrees of preference. You *also* get a rich implicit layer (shelved, "to-read", read-but-unrated) that the best models exploit alongside the stars.
3. **The catalog is large but near-static, and text-rich.** Millions of books, each with a title, description, author, series, genres/shelves, and often thousands of words of reviews. New books arrive slowly. This text is the cold-start lifeline and the fuel for the entire content/NLP and LLM half of the system.

The architecture that falls out of these facts is the **batch precompute-and-serve** pattern — the same one Amazon's classic item-to-item collaborative filtering used in 2003: compute the expensive part offline (similarities, embeddings, top-N lists), store the results, and serve them with a sub-millisecond key lookup. A nightly job retrains on the growing interaction log and refreshes recommendations; the serving path is a dumb, fast cache. If the nightly compute fails, you keep serving yesterday's list and no user notices.

This guide builds that system from baselines up to the LLM frontier, and is explicit throughout about *why batch is sufficient* and where the genuine state-of-the-art lives for this regime.

---

## 2. Explicit vs implicit feedback — the distinction that defines this domain

If you internalize one thing before modeling, make it this. The feedback type drives the loss function, the evaluation metric, and the failure modes.

| | **Implicit (music/streaming)** | **Explicit (Goodreads/books)** |
|---|---|---|
| Signal | Plays, skips, dwell — binary/continuous *engagement* | 1–5 star *ratings*, plus shelves/to-read |
| Can you tell "disliked" from "never seen"? | **No** — absence is ambiguous | **Yes** — a 2★ is a real negative |
| Natural objective | Confidence-weighted preference (ALS) or pairwise ranking (BPR) | Rating regression (RMSE/MAE) *and/or* ranking |
| Volume | Abundant, streaming | Sparse, bursty (people finish books occasionally) |
| Update cadence | Real-time | **Batch (nightly/weekly)** |
| Central pitfall | Selection bias from what was shown | **Missing Not At Random** (low ratings disproportionately absent) |

The practical consequences:

- **You can do true rating prediction.** Models like biased matrix factorization and SVD++ predict a star value, and the historical Netflix Prize lineage (scored on RMSE of 1–5 star predictions) applies directly. But — see Section 4 — **low RMSE does not guarantee a good top-N list**, so you usually *train* on ratings and *evaluate* (and often re-optimize) for ranking.
- **You have a companion implicit layer for free.** "Shelved but unrated" and "to-read" are abundant implicit signals. The fact that a user *chose to engage* with a book is itself information — which is exactly what **SVD++** formalizes (Section 5.3).
- **The big trap is MNAR.** Readers mostly rate books they already expected to like, so the *observed* rating distribution is upward-biased and unrepresentative of the full preference matrix. A model tuned only to minimize RMSE on observed stars can look great offline and recommend poorly in production. Section 4 and Section 11 treat this head-on.

---

## 3. The Goodreads data model and public datasets

### 3.1 The entities and the event

You're modeling four things plus an interaction:

- **Users**: stable ID, a history of ratings and shelf actions, optionally declared favorite genres/authors (onboarding).
- **Books (the item)**: book ID *and* the edition-independent **work ID** (the same novel across editions — deduplicate on this), title, description, author(s), **series**, publication year, language, average rating and the per-star ratings histogram.
- **Reviews**: full review text, rating, helpful votes, timestamp — a deep, underused text signal.
- **Shelves / tags ("popular_shelves")**: user-generated tags like `to-read`, `fantasy`, `favorites`, `book-club`. Noisy but high-signal; genres are a cleaned projection of these.
- **Interaction event**: `(user_id, book_id, rating ∈ {0..5, 0=unrated}, is_read, shelves, date)`. The "behavior chain" shelved → read → rated is itself informative.

### 3.2 Public datasets to practice on

| Dataset | Scale | Contents | Best for |
|---|---|---|---|
| **goodbooks-10k** | 10,000 books, **5,976,479** ratings, 53,424 users | `ratings.csv` (user,book,rating 1–5), `books.csv` metadata, `book_tags.csv`/`tags.csv` shelves, `to_read.csv` (implicit) | The clean "MovieLens of books" — start here for explicit-rating MF |
| **UCSD Goodreads "Book Graph"** (Wan & McAuley) | **~2.36M books**, ~876K users, **~229M interactions** (112M reads + 105M ratings), **~15M reviews** | Book metadata, authors, works, series, fuzzy genres, full review text, **per-sentence spoiler labels** (~1.3M reviews) | Realistic scale, reviews, shelves, KG edges. *Academic use only.* Per-genre subsets keep it tractable |
| **Amazon Reviews 2023 — Books** (McAuley Lab) | Whole corpus ~571M reviews / 48M items; Books is the largest category | 1–5 star reviews, text, helpful votes, verified flag, rich item metadata, "bought-together" graph | Product-review scale; note heavy positive skew (amplifies MNAR) |

Two cautions worth stating up front: goodbooks-10k is universally called "six million ratings" but the exact count is **5,976,479**; and "SVD" as used by the Surprise library and Simon Funk is **not** classical linear-algebra SVD — it's a regularized, SGD-trained low-rank factorization over observed ratings (Section 5.2). The UCSD dataset numbers above are from the authors' official page and are reliable.

### 3.3 Feature engineering that pays off for books

- **Per-book content document**: concatenate description + aggregated top-helpful reviews + title → the input to TF-IDF or an embedding model (Section 6).
- **Same-series / same-author flags**: near-deterministic positives. If a user loved book 3 of a series, book 4 is an easy win. Don't let a fancy model bury this.
- **Recency-weighted user taste vector**: exponentially decay older ratings so the profile tracks current taste — though decay is *gentle* here (months, not minutes).
- **Shelf-tag bag-of-words**: treat each book's shelves as a weighted document; strong genre signal.
- **Bias features**: per-user mean rating (some users rate everything 5★), per-book mean — these "baseline predictors" explain a large fraction of rating variance for almost free.

---

## 4. Framing the problem

Three framing decisions determine whether your offline numbers mean anything.

### 4.1 Rating prediction vs ranking — pick the metric that matches the product

There are two objectives, and they are *not* interchangeable:

- **Rating prediction (pointwise, RMSE/MAE).** Fit each observed `(user, book, rating)` and minimize squared/absolute error. Appropriate when you genuinely surface a predicted score ("we think you'd rate this 4.3★") or you're benchmarking in the Netflix/MovieLens tradition. The unrated majority is simply excluded from the loss.
- **Ranking (pairwise/listwise, Precision@K / NDCG / MAP).** Optimize the *order* of a top-N list. This is almost always the real product goal: "show me books to read next."

The trap that sinks many projects: **a model with excellent held-out RMSE can rank the catalog poorly.** RMSE weights all rated items equally and ignores the unrated majority entirely; a top-N list lives or dies on how it orders thousands of *unrated* candidates. The pragmatic stance for books: train on explicit ratings (the signal is real and valuable), but **evaluate and tune for ranking** (Section 11).

### 4.2 Missing Not At Random (MNAR) — the central statistical pitfall

Which ratings *exist* depends on their value. Readers self-select into books they expect to enjoy, so **low ratings are disproportionately missing** and the observed star distribution is upward-biased and unrepresentative. This was shown empirically by Marlin & Zemel (collecting ratings on *randomly chosen* items yields a very different distribution than self-selected ones), and Steck (KDD 2010) showed that standard RMSE-on-observed evaluation is itself biased under MNAR.

Why you should care as an engineer: a book model that minimizes RMSE on observed Goodreads stars never sees the (mostly negative) missing ratings and quietly inherits popularity/selection bias — it will over-recommend safe bestsellers. The defenses, in increasing order of effort:

1. **Evaluate with ranking metrics on a temporal split** (Section 11) rather than RMSE-on-observed alone.
2. **Inverse Propensity Scoring (IPS)**: weight each observed rating by `1 / P(it was observed)` to recover an unbiased risk estimate. Doubly-robust estimators (Wang et al., ICML 2019) combine IPS with imputation for lower variance.
3. **Beyond-accuracy metrics** (coverage, novelty) as guardrails so you notice when the model collapses to popularity.

### 4.3 Cold start is content's job

Because the catalog is text-rich, cold start is far more tractable than in domains with sparse item metadata:

- **New book (no ratings)**: build its content/embedding vector at ingestion and place it in the retrieval index immediately — no interactions required. Same-author/same-series flags give near-deterministic early recommendations.
- **New user (no history)**: onboard with a short "pick a few favorite genres/authors/books" flow; build an initial taste vector from those embeddings; let ratings accumulate and shift weight from content to collaborative.

The architectural payoff (Section 7): if users and items are represented by *features* (not just opaque IDs), both cold-start cases dissolve — features are always available, interactions fill in over time.

---

## 5. Collaborative filtering baselines (kNN, biased MF, SVD++, ALS, BPR)

Always build these first. They are cheap, strong, batch-native, and — per Rendle's 2020 result (Section 10.1) — a *well-tuned* matrix factorization is a bar that much fancier neural models routinely fail to clear. If your deep model can't beat tuned MF, something is wrong.

### 5.1 Neighborhood (kNN) CF

Predict from similar users or, better, **similar items**. Item-based kNN (Sarwar et al., 2001) is usually preferred: item–item similarities are more stable than user–user, precomputable offline (perfect for batch), and there are often fewer books than users. Mean-center ratings first to remove per-user/per-book rating-scale bias, and use Pearson or cosine similarity. This is exactly Amazon's classic item-to-item approach — and it gives you interpretable "because you liked X" explanations for free.

### 5.2 Biased matrix factorization (the workhorse)

Learn latent vectors `p_u` (user) and `q_i` (book) plus bias terms, trained by SGD over *observed* ratings only (this is "FunkSVD" / Surprise's `SVD`):

```
r̂_ui = μ + b_u + b_i + p_uᵀ q_i
```

`μ` is the global mean, `b_u` a user's tendency to rate high/low, `b_i` a book's overall quality/popularity. The biases alone (the "baseline predictor") explain a surprising amount of variance — never skip them.

```python
import torch, torch.nn as nn

class BiasedMF(nn.Module):
    """Biased matrix factorization for explicit 1-5 star rating prediction (FunkSVD-style)."""
    def __init__(self, n_users, n_items, dim=64, global_mean=3.9):
        super().__init__()
        self.user_f = nn.Embedding(n_users, dim)
        self.item_f = nn.Embedding(n_items, dim)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.tensor(global_mean), requires_grad=False)
        for emb in (self.user_f, self.item_f):
            nn.init.normal_(emb.weight, std=0.01)
        for b in (self.user_b, self.item_b):
            nn.init.zeros_(b.weight)

    def forward(self, u, i):
        dot = (self.user_f(u) * self.item_f(i)).sum(-1)
        return self.mu + self.user_b(u).squeeze(-1) + self.item_b(i).squeeze(-1) + dot

    def loss(self, u, i, r, l2=1e-5):
        pred = self.forward(u, i)
        mse = ((pred - r) ** 2).mean()                      # RMSE objective on observed ratings
        reg = l2 * (self.user_f(u).pow(2).sum() + self.item_f(i).pow(2).sum())
        return mse + reg
```

For a quick, batteries-included version use the **Surprise** library (`scikit-surprise`): it ships `SVD`, `SVDpp`, `NMF`, the kNN family, `BaselineOnly`, plus cross-validation and `GridSearchCV` with RMSE/MAE — purpose-built for explicit ratings on goodbooks-10k scale. Its limitation: no implicit-feedback or content features, and `SVDpp` is slow on large data.

### 5.3 SVD++ — fold in the implicit layer

Goodreads users shelve and mark "to-read" far more than they rate. **SVD++** augments biased MF with an implicit term: each user is *also* represented by the set `N(u)` of items they interacted with (rated or merely shelved), via implicit factor vectors `y_j`:

```
r̂_ui = μ + b_u + b_i + q_iᵀ ( p_u + |N(u)|^(−½) Σ_{j∈N(u)} y_j )
```

The mere fact that a user *chose to engage* with a book carries signal, independent of the star value. This is the single most natural way to combine explicit + implicit in one model, and it consistently improves over plain biased MF when shelving data is abundant — exactly the Goodreads situation.

### 5.4 ALS and BPR — when you treat signal as implicit / want ranking

- **ALS (Alternating Least Squares)** — the Hu–Koren–Volinsky implicit method splits each observation into a binary preference `p_ui` and a confidence `c_ui = 1 + α·r_ui`, then minimizes confidence-weighted squared error over **all** cells (observed and not), solving by alternating ridge regressions that parallelize cleanly. Use it when you model shelving/reading as implicit signal. (ALS also has an explicit mode over observed ratings only — "ALS" is the optimizer, the loss depends on feedback type.)
- **BPR (Bayesian Personalized Ranking)** — a pairwise objective: for each user, an interacted book should score above a non-interacted one. It optimizes ranking directly (`Σ ln σ(x̂_ui − x̂_uj)`), so it tends to beat pointwise RMSE models on top-N metrics.

```python
import torch.nn.functional as F

def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Pairwise ranking: the interacted (positive) book should outscore a sampled negative."""
    pos = (user_emb * pos_item_emb).sum(-1)
    neg = (user_emb * neg_item_emb).sum(-1)
    return -F.logsigmoid(pos - neg).mean()
```

The **implicit** library (`implicit`) provides fast Cython/GPU ALS, BPR, and Logistic MF and is the right tool when you treat the problem as implicit ranking at scale. Rule of thumb: **Surprise for explicit-rating RMSE work; implicit for implicit ranking at scale**; SVD++ or a hybrid when you want both signals in one model.

---

## 6. Content and NLP representations

Collaborative filtering is powerful but blind to any book nobody has rated yet — and the long tail of books is enormous. Content representations, built from the rich text Goodreads gives you, are how you recommend new and niche books, explain recommendations, and (Section 7) anchor the whole cold-start story. This is a bigger, more central part of a book system than of a music one, because books *are* text.

### 6.1 The TF-IDF baseline (build this, then beat it)

The classic content pipeline: build a per-book "content document" (description + aggregated top-helpful reviews + title), vectorize with **TF-IDF**, and compute item–item **cosine similarity**. Recommend nearest neighbors to a seed book, or to a *user profile vector* (the rating-weighted mean of the TF-IDF vectors of books they liked).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

def build_tfidf(books_df):
    docs = (books_df["description"].fillna("") + " "
            + books_df["title"].fillna("") + " "
            + books_df["shelves_text"].fillna(""))          # shelves as bag-of-words
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(docs)                              # sparse [n_books, vocab]
    return vec, X

def similar_books(X, book_idx, k=10):
    # linear_kernel == cosine when rows are L2-normalized (TfidfVectorizer normalizes by default)
    sims = linear_kernel(X[book_idx], X).ravel()
    top = np.argsort(-sims)[1:k + 1]                         # skip the book itself
    return list(zip(top, sims[top]))
```

Bigrams (`ngram_range=(1,2)`) help noticeably on book text. At catalog scale do **not** materialize the full N×N similarity matrix — use ANN (FAISS/hnswlib) over the vectors, or restrict candidates per genre. To build a *full* content vector, concatenate (weighted) blocks: text TF-IDF, multi-hot genres/shelves, author and series (one-hot or learned embedding), and a scaled publication-year feature. Same-series and same-author blocks carry outsized signal.

### 6.2 Text embeddings — the biggest single quality upgrade

Dense embeddings replace sparse TF-IDF and capture semantics TF-IDF misses ("dystopian future" ≈ "post-apocalyptic society"). The workhorses from the **sentence-transformers (SBERT)** family:

- **`all-MiniLM-L6-v2`** — 384-dim, fast, the standard cheap default; great for prototyping and large catalogs.
- **`all-mpnet-base-v2`** — 768-dim, higher quality, slower.
- **E5** (`intfloat/e5-*`) — strong, but *requires prefixes* (`"query: "` / `"passage: "`; books are passages).
- **BGE** (`BAAI/bge-*-en-v1.5`, `bge-m3`) — `bge-m3` handles up to **8192 tokens** (good for long description + reviews) and does dense+sparse+multi-vector retrieval.

Current (2025–2026) frontier embedding models worth knowing: **Qwen3-Embedding** (0.6B/4B/8B; top of multilingual MTEB, with Matryoshka user-selectable output dims 32–4096 and instruction-tunable inputs — the 0.6B is a strong drop-in upgrade over MiniLM), **NVIDIA NV-Embed-v2** (Llama-3.1-8B fine-tune), and API options like **Gemini Embedding**, **Cohere Embed**, and **OpenAI text-embedding-3-large**. Track the live **MTEB leaderboard**; the tasks that matter here are Retrieval (nDCG@10) and STS.

A practical recipe for a semantic book embedding:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")     # swap for Qwen3-Embedding / bge-m3 in prod

def book_embedding(description, top_reviews, w_desc=0.6, w_rev=0.4):
    desc_vec = model.encode(description, normalize_embeddings=True)
    if top_reviews:                                  # mean-pool a few top-helpful reviews
        rev_vecs = model.encode(top_reviews, normalize_embeddings=True)
        rev_vec = rev_vecs.mean(axis=0)
        vec = w_desc * desc_vec + w_rev * rev_vec
    else:
        vec = desc_vec
    return vec / (np.linalg.norm(vec) + 1e-9)        # re-normalize after fusing
```

Index these in a vector store (FAISS, hnswlib, Qdrant, pgvector) for "more like this" and as the candidate-generation backbone. One caveat: pretrained embeddings aren't tuned to your interaction data — for production, either **fine-tune contrastively** (co-read book pairs as positives) or feed the embeddings as *features* into a downstream ranker (Section 7, 9).

### 6.3 Mining the reviews: topics and aspect sentiment

Reviews capture reader-perceived themes and tone the publisher blurb omits — a signal unique to a domain this text-heavy.

- **Topic modeling.** LDA (Gensim/scikit-learn) yields per-book topic distributions usable as compact, interpretable content features. But LDA struggles on short texts — aggregate all of a book's reviews into one document. The modern alternative, **BERTopic** (embeddings → UMAP → HDBSCAN → c-TF-IDF), produces more coherent, stable topics on fragmented review text and reuses the embeddings from 6.2.
- **Aspect-Based Sentiment Analysis (ABSA).** Extract aspects (*plot*, *characters*, *pacing*, *writing style*, *ending*) and a sentiment per aspect — far richer than document-level sentiment. A book's aspect-sentiment vector ("strong world-building, weak pacing") becomes an item feature you can match to a reader's inferred aspect preferences, or use to re-rank which reviews you surface. Tools: PyABSA or a fine-tuned BERT/DeBERTa.

### 6.4 Knowledge graphs — books are natively relational

Books form a rich heterogeneous graph: *(book)–authored_by–(author)*, *(book)–in_series–(series)*, *(book)–has_genre–(genre)*, *(author)–co-author/influenced_by–(author)*. This captures **high-order connectivity** that flat content vectors and pure CF miss (liked book A → its author → another book → same genre → recommend). Two reference methods at the conceptual level:

- **RippleNet** (CIKM 2018) — propagates a user's historical preferences outward along KG edges over multi-hop "ripple sets," accumulating preference signal onto candidate items.
- **KGAT — Knowledge Graph Attention Network** (KDD 2019) — merges the user–item graph with the item KG into a Collaborative Knowledge Graph and runs attentive GNN propagation; the attention weights also yield interpretable "why" explanations, and it outperforms RippleNet and Neural FM.

KG/GNN methods are heavier (graph construction + GNN training) and best treated as a phase-2 upgrade. The UCSD dataset's authors/works/series files give you the edges almost for free, and **RecBole** ships KGAT/RippleNet/KGCN for fast benchmarking.

---

## 7. Hybrid models and cold start

The best book recommenders fuse collaborative and content signal. Burke's classic hybridization taxonomy maps cleanly onto the batch setting:

- **Weighted** — linear blend of CF score and content score. Simplest; tune or learn the weights. A fine first hybrid.
- **Switching** — choose the model by context: **content when history is sparse, CF when it's dense.** The cleanest cold-start switch.
- **Feature combination/augmentation** — merge content features and CF signals into one feature space, or feed one model's output into another (e.g., CF latent factors as features in a content model).
- **Stacking / learning-to-rank** — train a meta-model (**LightGBM/XGBoost** ranker) on the outputs of base CF + content models plus meta-features (user activity, item popularity, genre). **Feature-Weighted Linear Stacking** (Sill et al.; part of the 2nd-place Netflix solution) lets the blend *adapt* — lean on content for cold items, on CF for warm ones. This is the highest-leverage hybrid for a serious system.

### 7.1 LightFM — hybrid factorization that dissolves cold start

**LightFM** represents each user/item latent vector as the **sum of embeddings of its content features**, trained with **WARP loss** (Weighted Approximate-Rank Pairwise — samples negatives until it finds a rank-violator and optimizes top-k precision, usually beating BPR). Because an item *is* its features, LightFM can embed a **brand-new book with zero interactions** as long as it has a description/genre/author — exactly the property you want.

```python
from lightfm import LightFM
from lightfm.data import Dataset

# Build feature-aware mappings: every book carries genre/author/series features
ds = Dataset()
ds.fit(users=all_user_ids, items=all_book_ids,
       item_features=all_book_feature_tokens)            # e.g. "genre:fantasy", "author:1234", "series:88"

interactions, weights = ds.build_interactions(
    (u, b, r) for u, b, r in rating_triples)             # r as sample weight
item_features = ds.build_item_features(
    (b, feats) for b, feats in book_features.items())

model = LightFM(loss="warp", no_components=64)           # WARP optimizes top-k ranking
model.fit(interactions, item_features=item_features,
          sample_weight=weights, epochs=30, num_threads=8)

# A new book with features but no interactions still gets scored via its feature embeddings.
scores = model.predict(user_id, candidate_book_ids, item_features=item_features)
```

### 7.2 Two-tower and DropoutNet

A **two-tower** model whose *item tower ingests content features* (description embedding, genre, author, series) can embed any new book at inference without retraining — the standard modern cold-start retrieval architecture (built out in Section 8). **DropoutNet** takes a complementary angle: it trains a neural CF model with **dropout on the interaction input channel**, forcing the model to reconstruct relevance from content alone — explicitly optimizing for cold start rather than bolting on a content loss.

### 7.3 How cold start dissolves in a batch pipeline

The unifying idea: represent users and items by **features**, and both cold-start cases handle themselves.

- **New book**: gets a content/embedding vector at ingestion, enters the candidate pool via ANN similarity to warm books. The stacker down-weights the absent CF signal and leans on content; as ratings arrive, weight shifts smoothly toward CF — no architectural change.
- **New user**: a short genre/author onboarding builds an initial content profile vector; recommend by cosine to candidates; transition to CF as they rate. Keep onboarding short — over-asking causes drop-off.

This is why content/NLP isn't merely a baseline in the book domain — it's the structural backbone that keeps the whole system working at the edges, every single night.

---

## 8. Batch retrieval with a two-tower model + ANN

Once you outgrow per-user MF lookups, the scalable retrieval pattern is the **two-tower (dual-encoder)** model — and it is an unusually clean fit for books precisely *because* the system is batch.

### 8.1 Why two-tower, and why it's batch-perfect for books

Two independent encoders — a user tower and a book tower — map into a shared space where similarity is a dot product, enabling **Maximum Inner Product Search** via ANN. The decisive architectural point (and a second reason beyond Rendle's accuracy result in Section 10.1 to prefer dot-product over a learned-MLP similarity): **dot-product decomposes user and item into independent embeddings, so you can precompute every book embedding offline and serve retrieval as one user-vector lookup against an ANN index.** A learned-similarity MLP (NeuMF's MLP branch) does *not* decompose — you'd have to push every (user, book) pair through the network, which defeats batch ANN entirely.

The batch workflow maps onto the book domain's cadences beautifully:

1. **Item tower runs rarely.** The catalog grows slowly, so you re-embed books and rebuild the ANN index only when the catalog changes.
2. **User tower runs nightly.** Refresh each user's embedding from their updated history.
3. **Generate candidates** per user by querying the ANN index; store the top-N.

This is the same precompute-then-lookup logic Snap, Uber, and Google describe for production two-tower retrieval — just with comfortably slow refresh cadences because book taste is stable.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Tower(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class TwoTowerBooks(nn.Module):
    """User tower: history-derived features. Book tower: CONTENT features (so new books embed instantly)."""
    def __init__(self, user_dim, book_dim, out_dim=64):
        super().__init__()
        self.user_tower = Tower(user_dim, out_dim)
        self.book_tower = Tower(book_dim, out_dim)

    def forward(self, user_feats, book_feats):
        return self.user_tower(user_feats), self.book_tower(book_feats)

def sampled_softmax_loss(u, pos, temp=0.07):
    """In-batch negatives: other books in the batch are negatives for each user."""
    logits = (u @ pos.t()) / temp                  # [B, B], diagonal = positives
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)
```

### 8.2 ANN serving (built nightly, queried fast)

For a Goodreads-scale catalog (low millions of books), **HNSW on CPU** is typically more than sufficient — near-exact recall, no GPU needed. FAISS gives you HNSW plus IVF/PQ and GPU if you need billion-scale; ScaNN is Google's MIPS-optimized option.

```python
import faiss, numpy as np

book_vecs = book_vecs.astype("float32")            # [N, d], L2-normalized, computed in batch
index = faiss.IndexHNSWFlat(book_vecs.shape[1], 32)
index.metric_type = faiss.METRIC_INNER_PRODUCT
index.add(book_vecs)                               # build once per catalog refresh

def candidates_for_user(user_vec, k=500):
    scores, ids = index.search(user_vec.reshape(1, -1).astype("float32"), k)
    return ids[0], scores[0]                        # union these with content & popularity retrievers
```

As in any production system, use **several retrievers** unioned before ranking: two-tower ANN, content/embedding similarity (cold start + discovery), same-series/same-author rules, and a popularity/editorial fallback. Each covers a different failure mode.

---

## 9. Ranking, re-ranking, and beyond-accuracy objectives

Retrieval handed you a few hundred candidates per user. A ranker now orders them with rich cross-features, and a re-ranking step turns a ranked list into a good *reading experience*.

### 9.1 Learning-to-rank with gradient-boosted trees

In a batch setting with strong tabular features, **LightGBM/XGBoost with a ranking objective (LambdaMART)** is often the most cost-effective ranker — frequently competitive with deep models and far cheaper to train and serve. Feed it the retrieval scores plus cross-features: user-genre affinity × book-genre, user-author play count, same-series flag, predicted rating from MF, book popularity (log-compressed), recency, review sentiment aspects.

```python
import lightgbm as lgb

# Learning-to-rank: group = candidates per user; label = graded relevance (e.g., rating, or 1/0 read)
train = lgb.Dataset(X_train, label=y_train, group=group_sizes_train)
params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10, 20],
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=50)
ranker = lgb.train(params, train, num_boost_round=500)
scores = ranker.predict(X_candidates)              # order a user's candidate books
```

### 9.2 Re-ranking: diversity, novelty, and the bestseller trap

A list ordered purely by predicted relevance is often a *bad* recommendation: ten books by the same author, all safe bestsellers, nothing new. For books, **beyond-accuracy objectives are not optional** — recommending the same 50 bestsellers is accurate and useless. Apply, as a final policy step:

- **Maximal Marginal Relevance (MMR)** or **DPPs** for intra-list diversity (across authors, genres, eras).
- **Calibration** so the genre mix of the list roughly matches the user's actual reading history (don't serve 100% fantasy to a 70/30 fantasy/literary reader).
- **Novelty/serendipity** boosts so the long tail and pleasant surprises get exposure — the main lever against the popularity bias baked into the data (and into Goodreads' own co-read popularity weighting).

```python
import numpy as np

def mmr_rerank(cand_ids, relevance, book_vecs, k=20, lam=0.7):
    """lam→1 favors relevance; lam→0 favors diversity. Greedy MMR over candidate books."""
    selected, pool = [], list(range(len(cand_ids)))
    while pool and len(selected) < k:
        if not selected:
            best = max(pool, key=lambda i: relevance[i])
        else:
            sims = {i: max(book_vecs[i] @ book_vecs[j] for j in selected) for i in pool}
            best = max(pool, key=lambda i: lam * relevance[i] - (1 - lam) * sims[i])
        selected.append(best); pool.remove(best)
    return [cand_ids[i] for i in selected]
```

Because everything here runs in the nightly batch job, you can afford richer re-ranking logic than a latency-bound real-time system would tolerate — another quiet advantage of the batch regime.

---

## 10. The frontier: LLMs, RAG/conversational, and generative retrieval

This is where "most state of the art" lives for books — and the book domain is, in some ways, a *better* fit for LLM methods than streaming music is, because books are text and LLMs are text engines. But there is one hard rule that governs everything here: **never let the model freely generate book titles.** Open-domain LLMs hallucinate plausible-but-nonexistent books, authors, and ISBNs. Every design below exists to keep the LLM *grounded in your actual catalog*.

### 10.1 First, a load-bearing caveat: don't over-engineer the similarity function

Before reaching for anything fancy, internalize **Rendle et al., "Neural Collaborative Filtering vs. Matrix Factorization Revisited" (RecSys 2020)**: a properly tuned **dot product substantially outperforms the learned MLP similarity** of NCF/NeuMF. An MLP can in theory approximate a dot product, but learning even that simple function takes large capacity and lots of data — and still underperforms well-tuned MF. The takeaway for this whole section: complexity must *earn its place* against a strong MF/two-tower baseline. The frontier methods below are worth it for specific capabilities (natural-language queries, explanations, cold start), not as reflexive upgrades.

### 10.2 LLM-based recommendation

Two reference lines:

- **P5 — "Recommendation as Language Processing"** (RecSys 2022) converts *all* signals (interactions, metadata, reviews) into natural-language sequences and trains one text-to-text model across many recommendation tasks (rating prediction, sequential rec, explanation, review summarization) with a single language-modeling objective. It's the foundational "recommendation as a language task" paper.
- **"LLMs are Zero-Shot Rankers for Recommender Systems"** (Hou et al., ECIR 2024) frames recommendation as conditional ranking: the user's history is the condition, a *retrieved candidate set* is what the LLM ranks. GPT-4-class models can compete with trained models — **but** the paper documents three biases you must engineer around: LLMs struggle to perceive **interaction order/recency**, exhibit **position bias** (quality drops when the answer sits late in the candidate list), and over-rank **popular** items. Mitigations: shuffle candidate order and sample repeatedly (bootstrapping), and prompt explicitly about recency.

The hard production limits: **hallucinating out-of-catalog books** (the #1 risk — well-documented for titles/authors/citations), popularity bias, and inference cost/latency. The consensus pattern that survives contact with production is **LLM-as-reranker, not LLM-as-generator**: a trained recommender supplies a grounded candidate set, and the LLM only reorders and explains within it. Constrained decoding against the catalog reduces but doesn't fully eliminate hallucination, so grounding the *candidate set* is the real safeguard.

### 10.3 RAG and conversational recommendation ("books like X but darker")

This is the most practical, most batch-friendly way to put an LLM to work, and it's where a book recommender gets genuinely delightful. The architecture:

1. **Index the catalog (batch).** Embed each book (title + author + blurb + genres + top reviews — your Section 6.2 vectors) into a vector DB (pgvector, Qdrant, Milvus, FAISS). Precomputed; nothing online here.
2. **Retrieve (online, cheap).** Embed the user's natural-language query and/or their reading-history vector; run hybrid (dense + BM25) ANN to get top-K candidate books. Apply **metadata filters** ("but darker," "but shorter," "published after 2015") as predicates on the vector search.
3. **Generate/re-rank (online, grounded).** Hand the retrieved candidates to an LLM to re-rank, filter, and **explain** — grounded in the retrieved set, so it *cannot* recommend a book outside your catalog. This is the core anti-hallucination property of RAG for recommenders.

"Books like X but Y" decomposes naturally: retrieve neighbors of X in embedding space, then apply constraint Y via metadata filtering or LLM re-ranking. Recent (2024–2025) work confirms the production split — an external trained recommender supplies candidates; the LLM adapts to context, re-ranks, and narrates. Because catalog embeddings and the index are precomputed, the only online cost is the per-query retrieval + one grounded LLM call.

```python
# Sketch: grounded conversational book recommendation (RAG). The LLM never invents titles.
def recommend_conversational(query, user_history_vec, vector_db, llm, k=20):
    q_vec = embed(query)                                   # Section 6.2 embedding model
    blended = 0.5 * q_vec + 0.5 * user_history_vec         # mix intent + long-term taste
    candidates = vector_db.search(blended, k=k,            # ANN over the PRECOMPUTED catalog
                                  filters=parse_constraints(query))   # "after 2015", "< 400 pages"
    # The LLM only sees real books and may only choose/order/explain among them:
    prompt = build_grounded_prompt(query, user_history, candidates)
    return llm.rerank_and_explain(prompt, allowed_items=candidates)   # output constrained to candidates
```

### 10.4 Generative retrieval (TIGER / Semantic IDs) — and why books suit it

**TIGER — "Recommender Systems with Generative Retrieval" (NeurIPS 2023)** gives each item a **Semantic ID**: encode its content (title/description) with a text encoder (e.g., Sentence-T5), then quantize the embedding with an **RQ-VAE** into a short hierarchical code (coarse category → fine detail). Recommendation becomes a seq2seq task — a Transformer **autoregressively generates the next item's Semantic ID** — with no ANN index at serving time.

Why this is an unusually good fit for the *batch book* setting specifically: books have rich text ideal for Semantic-ID construction; the **slow-changing catalog means the RQ-VAE codebook and ID assignments can be recomputed in batch** when the catalog updates, rather than continuously; and new books get an ID from their content encoder (a cold-start advantage over pure-ID models). Caveats: generative retrieval can emit *invalid* ID tuples, so constrain decoding to valid catalog IDs (a prefix trie); and it's heavier than two-tower + ANN, so treat it as an advanced/optional track. Open-source RQ-VAE recommender implementations exist to start from.

### 10.5 How to actually adopt the frontier (pragmatic order)

1. Build the classic stack first (Sections 5–9). You need its candidates, features, and evaluation regardless.
2. Add a **RAG semantic-search path** for natural-language discovery and "more like X but Y." Highest delight-per-effort, fully grounded, batch-friendly.
3. Add an **LLM re-ranker** over your existing candidate sets for the top of the list, with explanations. Never a free-form generator.
4. Only if you have budget and the catalog's text richness justifies it: a **TIGER-style generative retriever** with batch-recomputed Semantic IDs, added as one more retriever in the union.

---

## 11. Evaluation: the part most projects get wrong

Offline evaluation is where book recommenders most often fool themselves. Three landmines, each with a concrete defense.

### 11.1 Metrics — match them to the objective

- **Rating prediction**: **RMSE / MAE** on observed held-out ratings. Use when the product surfaces a predicted score or you're benchmarking in the Netflix tradition.
- **Top-N ranking** (the usual real goal): **Precision@K, Recall@K, NDCG@K, MAP@K, MRR**. NDCG@K is the workhorse (rewards relevant items near the top via a positional discount). MRR focuses on the rank of the *first* relevant item.
- **Beyond-accuracy (essential for books)**: **coverage** (fraction of catalog ever recommended), **diversity** (intra-list variety across genres/authors), **novelty** (how non-popular the recommendations are), and **serendipity** (relevant-but-unexpected). These have been shown to drive real engagement; for books they are the main guardrail against a model that just reprints the bestseller list.

```python
import numpy as np

def ndcg_at_k(ranked_ids, relevant, k=10):
    dcg  = sum(1.0 / np.log2(rank + 2) for rank, i in enumerate(ranked_ids[:k]) if i in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

def catalog_coverage(all_recommended_ids, catalog_size):
    return len(set(all_recommended_ids)) / catalog_size
```

### 11.2 Split temporally, not randomly (this is not optional)

A batch system predicts the *future* from the *past*, so evaluate that way: **train on interactions before time T, test on interactions after T.** Most published work historically used random splits — one survey of 85 top-venue papers (2017–2019) found ~54% random, ~28% leave-one-out, and only ~12% split-by-timepoint — and **random splits leak future information and inflate results.** When researchers switched to proper temporal splits, measured accuracy dropped dramatically (one study saw sampled nDCG@10 fall 21.7–73.4%). If your offline numbers look too good, a leaky split is the first suspect.

### 11.3 Don't use sampled metrics; do account for MNAR

- **Full-catalog, not sampled.** Krichene & Rendle (KDD 2020) showed that evaluating against a small sampled set of negatives can be *inconsistent* with full metrics and can even reverse model rankings. Rank against the full catalog when feasible.
- **MNAR-aware evaluation.** Per Section 4.2, observed ratings are upward-biased. Prefer ranking metrics on the temporal split, report **popularity-stratified** results (how do you do on tail vs head?), and use **IPS-weighted** estimates if you can estimate observation propensities.

**Recommended recipe:** global temporal split → full-catalog Recall@K and NDCG@K → a beyond-accuracy panel (coverage, diversity, novelty, serendipity) → explicit popularity stratification. Only then, if relevant, RMSE/MAE for the rating-prediction head.

---

## 12. Batch serving architecture

The whole system is a periodic pipeline feeding a fast cache.

```
        NIGHTLY (or weekly) BATCH JOB                        ONLINE (per request)
  ┌───────────────────────────────────────────┐     ┌─────────────────────────────────┐
  │ 1. Pull new interactions (ratings/shelves) │     │ Request: user opens the app     │
  │ 2. Retrain MF / SVD++ / two-tower / ranker │     │   │                             │
  │ 3. Refresh user embeddings                 │ ──► │   ├─► KEY LOOKUP: top-N for user │
  │ 4. (Catalog change?) re-embed books,       │     │   │   from KV store (Redis/      │
  │     rebuild HNSW index, recompute Sem-IDs  │     │   │   DynamoDB) — sub-millisecond│
  │ 5. Generate top-N per user + re-rank       │     │   └─► (optional) RAG path for    │
  │ 6. Write top-N to KV store                 │     │       natural-language queries   │
  └───────────────────────────────────────────┘     └─────────────────────────────────┘
            │ precomputed recommendations                     │ new events logged back
            ▼                                                 ▼  (feed tomorrow's batch)
     KV store · Vector index · model artifacts ◄──────────────┘
```

Why this is the right architecture for books, point by point:

- **Precompute, then serve via lookup.** Generate top-N per user nightly, store in a key-value store, serve with a key lookup. This decouples (expensive) computation from (trivial) serving — sub-millisecond at request time. It's exactly Amazon's 2003 item-to-item design: similarities offline, fast lookup online.
- **Batch is *sufficient* because preferences are stable.** Book taste evolves over weeks; the catalog is near-static; reading is low-frequency and bursty. A nightly full retrain captures essentially all new signal with negligible staleness. There is no business value in updating a user's list the instant they finish a chapter.
- **Operational wins.** Cheaper compute (batch economies of scale), simpler ops, and **fault tolerance**: if tonight's job fails, you keep serving yesterday's recommendations and no user notices. The cache is a buffer against compute failures.
- **Contrast with streaming.** Real-time systems (Kafka/Flink, lambda/kappa, online-learned models) exist to chase *fast-changing intent* — news, short-video, live e-commerce sessions. They add major infrastructure and operational complexity that a stable-preference book recommender simply doesn't need. A light online layer for "the book you're looking at right now" is the most you'd add, and even that is usually overkill.
- **ANN engine.** For a low-millions catalog, **HNSW on CPU** (tune `efSearch`, `M`) is plenty; FAISS/ScaNN if you scale up. Built in batch, queried fast.

Recompute cadence rule of thumb: **retrain nightly or weekly** on the interaction log; **rebuild the book index only when the catalog meaningfully changes** (infrequent). Refresh user embeddings nightly.

---

## 13. A concrete build roadmap

A phased plan that always leaves you with a working, evaluable system.

**Phase 0 — Data + evaluation harness (week 1–2).** Load goodbooks-10k (then UCSD for scale). Build the **temporal-split, full-catalog** evaluation (NDCG/Recall + coverage/diversity) *before* modeling. Deduplicate editions on `work_id`.

**Phase 1 — Baselines (week 2–3).** Popularity, item-based kNN, and **biased MF** (Surprise `SVD`). Establish the RMSE and ranking bars. Ship item-item "readers also enjoyed."

**Phase 2 — Fold in implicit + tune (week 3–4).** **SVD++** to exploit shelves/to-read; or treat as implicit and use ALS/BPR via `implicit`. Tune hard — remember Rendle: this MF bar is what your neural models must beat.

**Phase 3 — Content + embeddings (week 4–6).** TF-IDF baseline, then SBERT/Qwen3 book embeddings + ANN index. Add review topics (BERTopic) and aspect sentiment as features. Cold-start coverage jumps.

**Phase 4 — Hybrid + ranker (week 6–8).** **LightFM** (WARP) or a two-tower model with a content item-tower; then a **LightGBM LambdaMART** re-ranker over unioned candidates, with MMR/calibration re-ranking. This is the full classic stack.

**Phase 5 — LLM/RAG frontier (week 8+).** A grounded **RAG** path for natural-language discovery and "like X but Y"; an **LLM re-ranker + explanations** over existing candidates (never free-form generation). Optional: TIGER-style Semantic IDs given the catalog's text richness.

**Throughout**: serve precomputed top-N from a KV store; retrain nightly; evaluate each phase against the previous one on the temporal split; watch popularity/coverage guardrails.

---

## 14. Reading list and primary sources

**Datasets:**
- goodbooks-10k: https://github.com/zygmuntz/goodbooks-10k
- UCSD Goodreads "Book Graph" (Wan & McAuley): https://mengtingwan.github.io/data/goodreads.html · https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
- Amazon Reviews 2023 (McAuley Lab): https://amazon-reviews-2023.github.io/

**Collaborative filtering (explicit + implicit):**
- Funk, *Netflix Update: Try This at Home* (FunkSVD, 2006): https://sifter.org/simon/journal/20061211.html
- Koren, Bell, Volinsky, *Matrix Factorization Techniques for Recommender Systems* (IEEE Computer, 2009): https://www2.seas.gwu.edu/~simhaweb/champalg/cf/papers/KorenBellKor2009.pdf
- Hu, Koren, Volinsky, *Collaborative Filtering for Implicit Feedback Datasets* (ICDM 2008): https://www.chrisvolinsky.com/publications/17546-collaborative-filtering-for-implicit-feedback-datasets
- Rendle et al., *BPR: Bayesian Personalized Ranking* (UAI 2009): https://arxiv.org/abs/1205.2618
- Libraries: Surprise — https://surpriselib.com/ · implicit — https://benfred.github.io/implicit/ · LightFM — https://making.lyst.com/lightfm/docs/

**Content, NLP, embeddings, KG:**
- Sentence-Transformers pretrained models: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
- Qwen3-Embedding (2025): https://arxiv.org/pdf/2506.05176 · MTEB leaderboard overview: https://modal.com/blog/mteb-leaderboard-article
- BERTopic vs LDA (2024–2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11906279/
- RippleNet (CIKM 2018): https://arxiv.org/abs/1803.03467 · KGAT (KDD 2019): https://arxiv.org/pdf/1905.07854
- Feature-Weighted Linear Stacking: https://arxiv.org/abs/0911.0460 · DropoutNet (NIPS 2017): https://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf

**Neural CF, two-tower, and the frontier:**
- He et al., *Neural Collaborative Filtering* (WWW 2017): https://arxiv.org/abs/1708.05031
- Rendle et al., *NCF vs Matrix Factorization Revisited* (RecSys 2020) — **read this**: https://arxiv.org/abs/2005.09683
- Two-tower retrieval at scale (Google Cloud): https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture
- Geng et al., *P5: Recommendation as Language Processing* (RecSys 2022): https://arxiv.org/abs/2203.13366
- Hou et al., *LLMs are Zero-Shot Rankers for Recommender Systems* (ECIR 2024): https://arxiv.org/abs/2305.08845 · code: https://github.com/RUCAIBox/LLMRank
- *LLM-Enhanced Recommender Systems: A Survey* (2024): https://arxiv.org/abs/2412.13432
- Rajput et al., *TIGER: Recommender Systems with Generative Retrieval* (NeurIPS 2023): https://shashankrajput.github.io/Generative.pdf · RQ-VAE recommender code: https://github.com/EdoardoBotta/RQ-VAE-Recommender

**Evaluation & methodology (avoid the landmines):**
- Marlin & Zemel, *Collaborative Filtering and the Missing-at-Random Assumption* (UAI 2009).
- Steck, *Training and Testing of Recommender Systems on Data Missing Not at Random* (KDD 2010).
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020).
- *A Critical Study on Data Leakage in Recommender System Offline Evaluation*: https://arxiv.org/abs/2010.11060
- Evaluating recommender systems (ranking metrics): https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems

**Real-world systems:**
- Linden, Smith, York, *Amazon.com Recommendations: Item-to-Item Collaborative Filtering* (IEEE 2003): https://dl.acm.org/doi/10.1109/MIC.2003.1167344
- *The history of Amazon's recommendation algorithm* (Amazon Science): https://www.amazon.science/the-history-of-amazons-recommendation-algorithm

---

### A final word on "state of the art" for this regime

For streaming music, state-of-the-art means real-time sequential models and generative recommenders fed by a constant event stream. For **books, the state of the art is a different shape**: a well-tuned collaborative core (heed Rendle — beat strong MF before adding neural complexity), a content/embedding layer that makes cold start a non-event, a hybrid ranker with beyond-accuracy re-ranking, and a **grounded LLM/RAG layer** that adds natural-language discovery and explanations without ever inventing a book that doesn't exist — all retrained nightly and served from a precomputed cache. The batch architecture isn't a compromise here; it's the correct answer to a domain where preferences are stable and feedback is explicit. The code scaffolds above give you a runnable starting point for every stage.






