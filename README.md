# Five-card draw CFR solver

Chance-sampled Counterfactual Regret Minimisation (CFR, with an optional CFR+ mode) for heads-up, fixed-limit five-card draw on reduced decks. This is the code behind our COMP3821 progression check (`progress_check.md`; figures in `figures/`, raw results in `results/`).

## The game

- Deck of 4 suits and the top `n_ranks` ranks (default 6, giving 9 T J Q K A, 24 cards).
- Ante 1 each. Fixed-limit betting with a bet of 1 before the draw and 2 after, and at most 2 bets or raises per round. P0 acts first in both rounds.
- One draw. P0 draws first, then P1. Each player sees the other's draw count, never the cards.
- Hand ranking follows our proposal's rule. High card, pair, two pair and three of a kind keep the standard order. Straight and above are re-ordered by rarity in the chosen deck, computed by enumeration. In the 20-card deck, flushes cannot occur, so the position of the flush there has no effect.

## Abstractions (what makes it solvable)

- **Discards.** A player chooses only *how many* cards to draw, from 0 to 3. Which cards go is fixed by `game.keep_cards`: paired cards are kept first, then four-card flush or straight draws, then the highest cards.
- **Cards.** Information sets use hand buckets, not exact cards. Before the draw, a bucket is the category, its main rank and a draw flag. After the draw, it is the category and its top two tie-break ranks. The post-draw bucket forgets the pre-draw hand, so the abstract game has *imperfect recall* (Waugh et al., 2009b).

## Layout

| File | Contents |
| --- | --- |
| `fcd/cards.py` | Decks, reduced-deck ranking, hand evaluator with tie-breaks |
| `fcd/game.py` | Betting rules, discard rule, card buckets, one sampled deal |
| `fcd/cfr.py` | Chance-sampled CFR / CFR+ trainer |
| `fcd/evaluate.py` | Random and greedy baselines, exact-over-actions EV, behaviour statistics |
| `run_experiments.py` | `train` and `analyse` commands |
| `make_plots.py` | Figures from `results/` |
| `tests/test_engine.py` | Evaluator and rules tests |

## Reproducing the results

Requires Python 3.10+ and matplotlib (for plots only).

```
python tests/test_engine.py
python run_experiments.py train --ranks 6 --iters 150000 --tag cfr24
python run_experiments.py train --ranks 6 --iters 150000 --plus --tag cfrplus24
python run_experiments.py train --ranks 5 --iters 100000 --tag cfr20
python run_experiments.py train --ranks 7 --iters 100000 --tag cfr28
for t in cfr24 cfrplus24 cfr20 cfr28; do python run_experiments.py analyse --tag $t; done
python make_plots.py
```

Each 24-card run takes about 15 minutes on one laptop core. Seeds are fixed, so the results reproduce exactly.
