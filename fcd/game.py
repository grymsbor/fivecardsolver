"""
Rules of heads-up, fixed-limit five-card draw, plus the discard and card
abstractions the solver uses.

Game flow (P0 acts first in both betting rounds, as the non-dealer does):
  deal 5 cards each -> betting round 1 -> P0 draws, then P1 draws
  (each sees the other's draw COUNT, never the cards) -> betting round 2
  -> showdown.

History string:  "<round 1 actions>/<draw counts>/<round 2 actions>"
  actions: k = check, b = bet, r = raise, c = call, f = fold
  e.g. "kbc/31/bc" = check, bet, call; P0 draws 3, P1 draws 1; bet, call.
"""

from collections import Counter
from dataclasses import dataclass

from .cards import make_deck


@dataclass(frozen=True)
class Config:
    # 6 ranks x 4 suits = 24-card deck (9 T J Q K A)
    n_ranks: int = 6
    n_suits: int = 4
    ante: int = 1
    # small bet before the draw, big bet after
    bet_sizes: tuple = (1, 2)
    # bets + raises allowed per round
    cap: int = 2
    draw_options: tuple = (0, 1, 2, 3)
    # "base" or "fine" card buckets (see pre_bucket)
    abstraction: str = "base"

    @property
    def deck(self):
        return make_deck(self.n_ranks, self.n_suits)

    def label(self):
        return f"{self.n_ranks * self.n_suits}-card"

    @property
    def fine(self):
        return getattr(self, "abstraction", "base") == "fine"


# ---------------- Betting ----------------

def legal_actions(round_hist, cap):
    bets = sum(a in "br" for a in round_hist)
    if round_hist and round_hist[-1] in "br":
        return ["f", "c", "r"] if bets < cap else ["f", "c"]
    return ["k", "b"]


def round_over(round_hist):
    return round_hist == "kk" or (round_hist != "" and round_hist[-1] in "cf")


def round_contributions(round_hist, size):
    level, contrib = 0, [0, 0]
    for i, a in enumerate(round_hist):
        p = i % 2
        if a in "br":
            level += size
            contrib[p] = level
        elif a == "c":
            contrib[p] = level
    return contrib


# ---------------- Discard abstraction ----------------
# Instead of 32 possible discards, a player only chooses HOW MANY cards to
# draw (0-3). Which cards go is fixed by this rule, which keeps the cards most
# likely to matter: paired ranks first, then four-card flush or straight draws,
# then the highest singletons.

def _draw_cards(deck, hand):
    """Four cards of a flush or straight draw, if the hand has no pair."""
    by_suit = Counter(deck.suit(c) for c in hand)
    suit, n = by_suit.most_common(1)[0]
    if n == 4:
        return [c for c in hand if deck.suit(c) == suit]
    ranks = {deck.rank(c): c for c in hand}
    for sset in deck.straights:
        inside = [ranks[r] for r in sset if r in ranks]
        if len(inside) == 4:
            return inside
    return None


def keep_cards(deck, hand, k):
    if k == 0:
        return list(hand)
    counts = Counter(deck.rank(c) for c in hand)
    if k == 1 and max(counts.values()) == 1:
        draw = _draw_cards(deck, hand)
        if draw:
            return draw
    order = sorted(hand, key=lambda c: (counts[deck.rank(c)], deck.rank(c)), reverse=True)
    return order[:5 - k]


# ---------------- Card abstraction ----------------
# Information sets are keyed on a hand bucket rather than the exact cards.
# Before the draw the bucket records the category, its main rank and whether
# the hand holds a four-card flush or straight draw. After the draw it records
# the category and its top two tie-break ranks. The post-draw bucket does not
# remember the pre-draw hand, so the abstract game has imperfect recall.
# The "fine" abstraction (used to test abstraction sensitivity) adds one more
# tie-break rank to each bucket.

CAT_CODE = {"High card": "H", "One pair": "P", "Two pair": "2P", "Three of a kind": "T",
            "Straight": "S", "Flush": "F", "Full house": "FH", "Four of a kind": "Q",
            "Straight flush": "SF"}


def _tiebreak_ranks(deck, hand, cat):
    ranks = [deck.rank(c) for c in hand]
    if cat in ("Straight", "Straight flush"):
        return [deck.straights[frozenset(ranks)]]
    cnt = Counter(ranks)
    return sorted(cnt, key=lambda r: (cnt[r], r), reverse=True)


def pre_bucket(deck, hand, fine=False):
    cat = deck.category(hand)
    tb = _tiebreak_ranks(deck, hand, cat) + [0]
    flag = "+d" if cat == "High card" and _draw_cards(deck, hand) else ""
    if fine:
        return f"{CAT_CODE[cat]}{tb[0]}.{tb[1]}{flag}"
    return f"{CAT_CODE[cat]}{tb[0]}{flag}"


def post_bucket(deck, hand, fine=False):
    cat = deck.category(hand)
    tb = _tiebreak_ranks(deck, hand, cat) + [0, 0]
    if fine:
        return f"{CAT_CODE[cat]}{tb[0]}.{tb[1]}.{tb[2]}"
    return f"{CAT_CODE[cat]}{tb[0]}.{tb[1]}"


# ---------------- One sampled deal ----------------

class Deal:
    """A shuffled deck with everything the tree walk needs, computed lazily:
    each player's final hand and score for every pair of draw counts."""

    def __init__(self, cfg, order):
        self.cfg, self.deck = cfg, cfg.deck
        self.order = order
        self.hands = [order[0:5], order[5:10]]
        self._pre = [pre_bucket(self.deck, h, cfg.fine) for h in self.hands]
        self._final = {}

    def pre(self, p):
        return self._pre[p]

    def final(self, draws):
        """(final hands, scores, post buckets) for draw counts (k0, k1).
        P0 draws first, so P1's replacements start after P0's."""
        if draws not in self._final:
            k0, k1 = draws
            h0 = keep_cards(self.deck, self.hands[0], k0) + self.order[10:10 + k0]
            h1 = keep_cards(self.deck, self.hands[1], k1) + self.order[10 + k0:10 + k0 + k1]
            scores = (self.deck.score(h0), self.deck.score(h1))
            buckets = (post_bucket(self.deck, h0, self.cfg.fine), post_bucket(self.deck, h1, self.cfg.fine))
            self._final[draws] = ((h0, h1), scores, buckets)
        return self._final[draws]


def fold_utility(cfg, r1, r2, folder):
    """Payoff to P0 when `folder` folds."""
    c = contributions(cfg, r1, r2)
    return -c[0] if folder == 0 else c[1]


def contributions(cfg, r1, r2):
    a = round_contributions(r1, cfg.bet_sizes[0])
    b = round_contributions(r2, cfg.bet_sizes[1])
    return [cfg.ante + a[0] + b[0], cfg.ante + a[1] + b[1]]


def showdown_utility(cfg, r1, r2, scores):
    # equal at showdown
    stake = contributions(cfg, r1, r2)[0]
    if scores[0] > scores[1]:
        return stake
    if scores[0] < scores[1]:
        return -stake
    return 0
