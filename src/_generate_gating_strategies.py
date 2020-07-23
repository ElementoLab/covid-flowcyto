"""
A programatic way of specifying the gating strategies used for the single cell analysis
"""

import json

from src.conf import GatingStrategy

# Gating strategy as list of Tuple[Channel, position of population to select]
# # the populations are chosen from the GMM that maximizes a sillhouette score
gating_strategies = {
    "WB_Memory": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("CD3(FITC-A)", -1)]
    ),
    "WB_IgG_IgM": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("CD19(Pacific Blue-A)", -1)]
    ),
    "WB_Checkpoint": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("CD3(FITC-A)", -1)]
    ),
    "WB_Treg": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("sCD3(FITC-A)", -1)]
    ),
    "WB_NK_KIR": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("CD56(Pacific Blue-A)", -1),]
    ),  # ("CD3", 0),
    "PBMC_MDSC": GatingStrategy(
        [("Viability(APC-R700-A)", 0), ("CD16(BV605-A)", -1)]
    ),
    "WB_T3": GatingStrategy(
        [
            ("Viability(APC-R700-A)", 0),
            ("sCD3(PE-Cy7-A)", -1),
            ("CD4(BV605-A)", -1),
        ]
    ),
}

json.dump(
    gating_strategies,
    open("metadata/gating_strategies.single_cell.json", "w"),
    indent=4,
)
