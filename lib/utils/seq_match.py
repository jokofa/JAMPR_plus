#
from typing import List
import itertools as it
import difflib
import numpy as np


def to_string_seq(tour: List[int]) -> str:
    """Convert tour to a string sequence."""
    return ' '.join(str(e) for e in tour)


def plan_to_string_seq(plan: List[List[int]]) -> str:
    """Convert tour plan represented as list of lists to a string sequence."""
    return ' '.join(str(t) for t in it.chain.from_iterable(plan))


def plan_to_string_text(plan: List[List]) -> str:
    """Convert tour plan represented as list of lists to a
    multiline text string with one line per list."""
    return "\n".join(' '.join(str(e) for e in t) for t in plan)


def get_similarity_scores(anchor: str,
                          candidates: List[str],
                          ) -> List[float]:
    """Get similarity score for each candidate
    compared to the anchor sequence."""
    matcher = difflib.SequenceMatcher(isjunk=lambda x: x == " ", a=anchor)
    scores = []
    for c in candidates:
        matcher.set_seq2(c)
        scores.append(matcher.ratio())
    return scores


def get_most_diverse_idx(anchor: str,
                         candidates: List[str],
                         k: int = 1) -> List[int]:
    """Get the idx of the 'k' candidates which are
    the most diverse from the anchor sequence."""
    return np.argsort(get_similarity_scores(anchor, candidates))[:k]


def rm_common_subsequences(plan_a: List[List],
                           plan_b: List[List],
                           k: int = 1,
                           tau: float = 0.5,
                           ) -> List[List]:
    """Remove the 'k' subsequences from plan_b which have
    the most common elements with subsequences in plan_a."""
    scores = []
    b_strings = [to_string_seq(t) for t in plan_b if len(t) > 0]
    for tour_a in plan_a:
        if len(tour_a) == 0:
            continue
        a = to_string_seq(tour_a)
        scores.append(get_similarity_scores(a, b_strings))
    scores = np.array(scores).reshape(-1)
    idx = (-scores.reshape(-1)).argsort()[:k]   # -scores to sort in descending order
    # check threshold
    rm_idx = idx[scores[idx] >= tau]
    # always remove at least one subsequence
    if len(rm_idx) == 0:
        rm_idx = np.array([idx[0]])
    rm_idx = rm_idx % len(b_strings)
    return [t for i, t in enumerate(plan_b) if i not in rm_idx]


# ============= #
# ### TEST #### #
# ============= #
def create_plan(n, k):
    assert (n-5 > k) and (k > 2)
    range_n = np.arange(5, n)
    cuts = np.random.choice(range_n[1:-1], k-2, replace=False)
    cuts.sort()
    cuts = [0] + cuts.tolist() + [n]
    plan = np.array(range_n).copy()
    np.random.shuffle(plan)
    return [list(range(5))] + [plan[s:e].tolist() for s, e in zip(cuts[:-1], cuts[1:])]


def _test1():
    np.random.seed(1)

    N = 40
    K = 7
    C = 5

    plan_a = create_plan(N, K)
    plan_b = create_plan(N, K)
    print(plan_a)
    print(plan_b)

    plan_a = plan_to_string_seq(plan_a)
    plan_b = plan_to_string_seq(plan_b)
    print(plan_a)
    print(plan_b)

    candidate_plans = [create_plan(N, K) for _ in range(C)]
    print(candidate_plans)

    md_idx = get_most_diverse_idx(plan_a, [plan_to_string_seq(p) for p in candidate_plans], 3)
    print(f"most div: {md_idx}")
    print(candidate_plans[md_idx[0]])


def _test2():
    import numpy as np
    np.random.seed(1)

    N = 30
    K = 6

    plan_a = create_plan(N, K)
    plan_b = create_plan(N, K)
    print(plan_a)
    print(plan_b)

    plan_b_new = rm_common_subsequences(plan_a, plan_b, k=2)
    print(plan_b_new)

