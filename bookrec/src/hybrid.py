"""
Hybrid layer — blend collaborative + content, and handle cold start.

Two strategies (guide Section 7):
  * 'weighted'  : score = alpha * CF + (1 - alpha) * content   (both rescaled to [0,1])
  * 'switching' : use content when the user has < min_ratings history, else CF
                  (the cleanest cold-start switch)

The blend is what lets a brand-new user (content/onboarding) and a
brand-new book (content vector, no CF signal) both still get recommended.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cf_scores_to_unit(cf_scores: pd.Series, scale=(1.0, 5.0)) -> pd.Series:
    lo, hi = scale
    # CF models emit star-scale predictions; hybrid blending needs the same
    # 0..1 scale used by the content cosine scores.
    return ((cf_scores - lo) / (hi - lo)).clip(0, 1)


class HybridRecommender:
    def __init__(self, cf_model=None, content_model=None,
                 strategy: str = "weighted", alpha: float = 0.6,
                 min_ratings_for_cf: int = 5):
        self.cf = cf_model
        self.content = content_model
        self.strategy = strategy
        self.alpha = alpha                       # weight on CF in 'weighted' mode
        self.min_ratings_for_cf = min_ratings_for_cf

    def score(self, user_id, user_ratings: pd.DataFrame, candidate_ids) -> pd.Series:
        """Return a blended score per candidate book_id (higher = better)."""
        # Materialize once so every branch can reindex back to the same order.
        candidate_ids = list(candidate_ids)
        n_hist = len(user_ratings)

        content_scores = (self.content.score_user(user_ratings, candidate_ids)
                          if self.content is not None
                          else pd.Series(0.0, index=candidate_ids))

        # cold user with no usable CF signal -> content only
        # This switch is the main guardrail against recommending from a CF model
        # before the user has enough history to be comparable to neighbors.
        cf_available = self.cf is not None and n_hist >= self.min_ratings_for_cf
        if not cf_available:
            return content_scores.reindex(candidate_ids).fillna(0.0)

        cf_raw = self.cf.predict_for_user(user_id, candidate_ids)
        cf_unit = _cf_scores_to_unit(cf_raw)

        if self.strategy == "switching":
            return cf_unit.reindex(candidate_ids).fillna(0.0)

        # weighted blend (default)
        c = content_scores.reindex(candidate_ids).fillna(0.0)
        f = cf_unit.reindex(candidate_ids).fillna(0.0)
        blended = self.alpha * f + (1 - self.alpha) * c
        return blended.rename("hybrid_score")
