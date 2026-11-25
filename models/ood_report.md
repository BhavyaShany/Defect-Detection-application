# Input-check report

Model fingerprint: `d36c9f7d8016` · layout: official · fit images: 2500 · limit = 99.5th percentile of 994 held-out photos (squared distance 1552.5) · grayscale training data: True

| Image set | Images | Flagged (either check) % | by distance % | by colour % |
|---|---|---|---|---|
| Held-out TEST photos (should be ~0 %) | 651 | 1.4 | 1.4 | 0.0 |
| Synthetic: color noise (should be 100 %) | 10 | 100.0 | 100.0 | 100.0 |
| Synthetic: gray noise (should be 100 %) | 10 | 100.0 | 100.0 | 0.0 |
| Synthetic: flat image (should be 100 %) | 10 | 100.0 | 100.0 | 0.0 |
| Synthetic: gradient (should be 100 %) | 10 | 100.0 | 100.0 | 0.0 |
| Synthetic: dark gradient (color) (should be 100 %) | 10 | 100.0 | 100.0 | 100.0 |
| YOUR negatives: not_casting (should be 100 %) | 113 | 100.0 | 100.0 | 42.5 |

Interpretation: the first row is the price of the check (genuine casting photos sent to manual handling); the other rows are how many non-casting images are rejected. Images that closely resemble the training photos (e.g. a different metal part under the same lighting) may still pass - test with your own negatives.