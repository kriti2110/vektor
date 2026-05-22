"""Cross-encoder reranker fine-tuning loop.

╔══════════════════════════════════════════════════════════════════════════╗
║   ⚠️  STUB. Kriti — this is the "self-improving" part of the project.    ║
║                                                                          ║
║   Spec: TODO_YOU_BUILD.md §3                                             ║
║   Tests: tests/test_train.py (TBD — you'll write these too)              ║
║                                                                          ║
║   Required reading before coding:                                        ║
║     • Joachims 2005 on position-bias correction in click data            ║
║     • Sentence-Transformers CrossEncoder fine-tuning docs                ║
║                                                                          ║
║   Suggested order:                                                       ║
║     1. Write synthetic click generator first (scripts/synth_clicks.py)   ║
║     2. Get training loop working on synthetic data, hit any uplift       ║
║     3. Switch to FeedbackStore data when API is live                     ║
║     4. Add position-bias correction (IPS or PBM)                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from pathlib import Path

from vektor.rerank.feedback import FeedbackStore


def train_reranker(
    feedback_store: FeedbackStore,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_dir: Path = Path("./models/reranker"),
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    eval_split: float = 0.1,
) -> dict:
    """Fine-tune a cross-encoder on click-feedback triples.

    Returns a dict of eval metrics: {"mrr@10": float, "ndcg@10": float, ...}

    TODO(Kriti) — implement. See TODO_YOU_BUILD.md §3.

    Suggested approach:
      1. Pull (query, positive, negative) triples from feedback_store.iter_training_triples()
      2. Hold out eval_split for evaluation.
      3. Build sentence_transformers.CrossEncoder(base_model).
      4. Use sentence_transformers.cross_encoder.evaluation.CERerankingEvaluator
         to track MRR during training.
      5. Train with MultipleNegativesRankingLoss or BCEWithLogitsLoss.
      6. Save best checkpoint to output_dir; return eval metrics.
    """
    raise NotImplementedError("Kriti — implement reranker fine-tune. See TODO_YOU_BUILD.md §3")
