"""Baselines sharing PriER's blocks, anchor prior, reviewer and token budget."""

BASELINES = ("risk", "ut", "booster", "alper", "acc", "kwikbucks", "kwikbucks_budget", "pivot")


def run_baseline(method, block, reviewer, budget, seed):
    """Run one baseline on one block.

    ``block`` provides ``anchor`` (prior matrix), ``cosine``, ``texts``,
    ``quotes``, ``channel`` and ``pair_weight``; ``reviewer`` is a
    :class:`prier.reviewer.BlockReviewer` for that block.
    """
    if method in ("risk", "ut"):
        from .simple import run
        return run(method, block.anchor, block.channel, block.quotes, reviewer, budget, block.pair_weight)
    if method == "booster":
        from .booster import run
        return run(block.anchor, block.channel, block.quotes, reviewer, budget, block.pair_weight)
    if method == "alper":
        from .alper import run
        return run(block.cosine, block.texts, reviewer, budget, seed)
    if method == "acc":
        from .acc import run
        return run(block.anchor, reviewer, budget, seed)
    if method == "kwikbucks":
        from .kwikbucks import run_original
        return run_original(block.cosine, reviewer, budget, seed)
    if method == "kwikbucks_budget":
        from .kwikbucks import run_budgeted
        return run_budgeted(block.cosine, reviewer, budget, seed)
    if method == "pivot":
        from .kwikbucks import run_pivot
        return run_pivot(block.cosine, block.anchor, seed, block.pair_weight)
    raise ValueError(f"unknown baseline: {method}")
