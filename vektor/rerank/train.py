"""Cross-encoder reranker fine-tuning loop.

Pulls (query, positive_doc, negative_doc) triples from the FeedbackStore via
skip-above heuristic (clicked docs are positives; docs shown above a click
that were not clicked are negatives). Fine-tunes a CrossEncoder with binary
cross-entropy, holds out a validation split, returns MRR@10 before/after.

For development before real traffic exists, generate triples first with
`python scripts/synth_clicks.py`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from vektor.rerank.feedback import FeedbackStore


def _evaluate_mrr(model, eval_pairs: list[tuple[str, str, str]], k: int = 10) -> float:
    """eval_pairs: [(query, positive_doc, negative_doc), ...]
    Score (q, pos) and (q, neg); MRR rewards models that rank pos > neg.
    """
    if not eval_pairs:
        return 0.0
    mrr_sum = 0.0
    for q, pos, neg in eval_pairs:
        scores = model.predict([(q, pos), (q, neg)], show_progress_bar=False)
        # rank pos: 1 if pos_score > neg_score, else 2
        rank = 1 if float(scores[0]) > float(scores[1]) else 2
        mrr_sum += 1.0 / rank
    return mrr_sum / len(eval_pairs)


def train_reranker(
    feedback_store: FeedbackStore,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    output_dir: Path = Path("./models/reranker"),
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    eval_split: float = 0.1,
    max_train_samples: int | None = None,
    seed: int = 42,
) -> dict:
    """Fine-tune a cross-encoder on click-feedback triples.

    Returns {"mrr_before": float, "mrr_after": float, "n_train": int, "n_eval": int}.
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    rng = random.Random(seed)

    # 1. Pull triples
    triples = list(feedback_store.iter_training_triples())
    if not triples:
        raise ValueError(
            "no training triples in feedback store — run scripts/synth_clicks.py "
            "first, or collect real click data via the /feedback endpoint"
        )
    rng.shuffle(triples)
    if max_train_samples is not None:
        triples = triples[: max_train_samples + max(1, int(len(triples) * eval_split))]

    n_eval = max(1, int(len(triples) * eval_split))
    eval_triples = triples[:n_eval]
    train_triples = triples[n_eval:]

    # 2. Build training examples — each triple → two labeled pairs
    train_examples = []
    for q, pos, neg in train_triples:
        train_examples.append(InputExample(texts=[q, pos], label=1.0))
        train_examples.append(InputExample(texts=[q, neg], label=0.0))

    # 3. Baseline MRR
    model = CrossEncoder(base_model, num_labels=1)
    mrr_before = _evaluate_mrr(model, eval_triples)

    # 4. Fine-tune
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup_steps = math.ceil(len(train_loader) * epochs * 0.1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=str(output_dir),
        show_progress_bar=False,
    )

    # 5. Post-training MRR
    mrr_after = _evaluate_mrr(model, eval_triples)

    return {
        "mrr_before": mrr_before,
        "mrr_after": mrr_after,
        "uplift": mrr_after - mrr_before,
        "n_train": len(train_triples),
        "n_eval": len(eval_triples),
        "model_path": str(output_dir),
    }


if __name__ == "__main__":
    import json

    import click

    from vektor.config import settings

    @click.command()
    @click.option("--epochs", default=3, type=int)
    @click.option("--batch-size", default=16, type=int)
    @click.option("--max-samples", default=None, type=int)
    def main(epochs: int, batch_size: int, max_samples: int | None) -> None:
        store = FeedbackStore(settings.feedback_db_path)
        result = train_reranker(
            store,
            epochs=epochs,
            batch_size=batch_size,
            max_train_samples=max_samples,
        )
        store.close()
        click.echo(json.dumps(result, indent=2))

    main()
