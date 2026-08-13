"""
Raw node-tag -> chemical element mapping for THIS project's MUTAG.txt
(the weihua916/powerful-gnns GIN-paper distribution, not the raw TU
Dortmund node_labels.txt -- these two use different tag orderings and
should not be assumed interchangeable).

Derivation (not assumed -- computed from the actual uploaded file):
    tag | count | max_deg | avg_deg | atom | reasoning
    ----+-------+---------+---------+------+---------------------------------
     2  | 2395  |    4    |  2.44   |  C   | dominant count, valence up to 4
     6  |  593  |    2    |  1.04   |  O   | mostly deg-1 (C=O / N-O), some deg-2
     5  |  345  |    3    |  2.75   |  N   | valence up to 3 fits nitro-N / ring N
     1  |   23  |    1    |  1.00   |  Cl  | most common halogen (matches lit.)
     3  |   12  |    1    |  1.00   |  Br  | second most common halogen
     0  |    2  |    1    |  1.00   |  F   | rare terminal substituent
     4  |    1  |    1    |  1.00   |  I   | rarest

Cross-checks that confirmed this is the genuine, correctly-parsed file:
  - total_nodes == 3371, matching the documented official MUTAG node count.
  - graph_labels == {0: 63, 2: 125}, matching the documented 63/125 class
    split for the standard 188-graph MUTAG benchmark.

Caveat: tag 0 vs tag 4 (F vs I) rests on only 2 vs 1 atoms in the whole
dataset -- both are always degree-1, so degree alone can't disambiguate
them. This assignment follows typical relative abundance (F seen slightly
more often than I in this compound class) but is the one part of this
table that is inference rather than a direct structural proof. It affects
at most 3 atoms total across the whole dataset and is very unlikely to
matter for a demo, but flag it if you can independently confirm either way.

Since gin/data.py builds tag_to_idx from sorted(all_tags) and the tags in
this file are exactly the contiguous set {0..6}, one-hot index == raw tag.
"""

ATOM_MAP = {0: "F", 1: "Cl", 2: "C", 3: "Br", 4: "I", 5: "N", 6: "O"}
ELEMENT_TO_TAG = {v: k for k, v in ATOM_MAP.items()}

NUM_TAGS = len(ATOM_MAP)  # 7, matches in_dim the model was trained with

# Graph-label convention (see docstring in train_and_export.py for the
# provenance of this assignment): sorted({0, 2}) -> {0: 0, 2: 1}, and the
# well-documented 125/63 MUTAG split has raw label 2 (n=125) = mutagenic.
CLASS_NAMES = {0: "Non-mutagenic", 1: "Mutagenic"}
