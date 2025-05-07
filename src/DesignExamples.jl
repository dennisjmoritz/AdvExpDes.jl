

using DataFrames, StatsModels
#Todo: Determine if this is useful
# — Test Values / Contrast Matrix for Lenth’s Pseudo‐standard Error —─────────────
# C from Lenth (1989) example
const C = [
    0   0   1;
   -1   0   1;
   -1   0   1;
   -1   0   1;
   -1   0   1;
    0   1  -1
]


# — NIST 2³ Factorial + Center Points (12 runs) —────────────────────────────────
# Model matrix from 
# https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.html  
# D‐optimal design metrics: D = 0.6825575, A = 2.2, G = 1, I = 4.6625
const ModMat = [
  -1  -1  -1;
  -1  -1  +1;
  -1  +1  -1;
  -1  +1  +1;
   0  -1  -1;
   0  -1  +1;
   0  +1  -1;
   0  +1  +1;
  +1  -1  -1;
  +1  -1  +1;
  +1  +1  -1;
  +1  +1  +1
]

const ModMat_df = DataFrame(
  y  = repeat([2], 12),
  x1 = repeat([-1, 0, +1], inner = 4),
  x2 = repeat(repeat([-1, +1], inner = 2), 3),
  x3 = repeat([-1, +1], 6)
)

# — Two‐Factor Small Exact Response‐Surface Design (6 runs) —────────────────────
# From “Small Exact Response Surface Designs” (J. PSS 2003)
# D = 42.3123, A = 23.7678, G = 53.6542, I = 0.203900
const design6_1 = [
   1.000000   1.000000;
  -1.000000   1.000000;
  -1.000000  -1.000000;
   1.000000  -0.394449;
   0.394449  -1.000000;
  -0.131483   0.131483
]


# — SKPR Design (8 runs) —───────────────────────────────────────────────────────
# From “skpr” JSS article, page 6
const skpr_design = [
  -1  -1  -1   1   1   1;
  -1  -1  -1   1   1   1;
  -1   1   1  -1  -1   1;
  -1   1   1  -1  -1   1;
   1  -1   1  -1   1  -1;
   1  -1   1  -1   1  -1;
   1   1  -1   1  -1  -1;
   1   1  -1   1  -1  -1
]


