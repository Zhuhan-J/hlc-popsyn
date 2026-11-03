# Toy example

I use this example to run the full workflow on a small table of random integer codes. The table
has no connection to any real data. It only has the column layout that `hm-popsyn-run` expects
(see [Input format](../README.md#input-format) in the main README). Run the commands from the
repository root after `pip install -e ".[dev]"`.

## 1. Make the toy data

```bash
python examples/make_toy_data.py
```

```text
Wrote 1287 individuals in 500 households to examples/toy_merged.csv
```

The script uses a fixed seed (`--seed 2018`) and 500 households by default, so rerunning it
rewrites the committed `examples/toy_merged.csv` (about 35 KB) with identical content. Household
sizes are drawn from 1 to 5, and the first household always has one member, so shifting each
column to a minimum of 1 leaves the sizes unchanged. All other attributes are uniform random
codes.

## 2. Fit the model and generate a synthetic population

```bash
hm-popsyn-run --input examples/toy_merged.csv --output outputs/toy_result.npz \
    --G 4 --M 6 --n-households 1000 --n-restarts 2 --max-iter 500 --tol 1e-6 --seed 1
```

The command fits EM with 2 restarts, generates 1000 households with the rejection step and writes
`outputs/toy_result.npz`. It took about 15 seconds in my test run. It prints only the rejection
diagnostics, and the first block of the output was:

```text
[REJECT-First-pass 0] Applying rejection for size=2 households...
      [ALPHA] Scaling factor M: raw=1.308 -> applied=1.308
      [ALPHA] Households: 418, accept-ready (alpha>0): 418 (100.0%)

[REJECT-First-pass 0] Applying rejection for size=3 households...
      [ALPHA] Scaling factor M: raw=2.902 -> applied=2.902
      [ALPHA] Households: 267, accept-ready (alpha>0): 267 (100.0%)

[REJECT-First-pass 0] Applying rejection for size=4 households...
      [ALPHA] Scaling factor M: raw=2.400 -> applied=2.400
      [ALPHA] Households: 222, accept-ready (alpha>0): 212 (95.5%)

[REJECT-First-pass 0] Applying rejection for size=5 households...
      [ALPHA] Scaling factor M: raw=1.366 -> applied=1.366
      [ALPHA] Households: 173, accept-ready (alpha>0): 168 (97.1%)

====================================================================================================
[DIAG-REJECT] First-pass 0: REJECTION SAMPLING VERIFICATION
====================================================================================================

1. HOUSEHOLD SIZE DISTRIBUTION (before vs after rejection):
  Size       Before        After      Dropped      Drop%
----------------------------------------------------------------------
     1          420          420            0       0.0%
     2          418          318          100      23.9%
     3          267           80          187      70.0%
     4          222           91          131      59.0%
     5          173          130           43      24.9%

2. PER-SIZE ACCEPTANCE SUMMARY:
  Size     Proposed     Accepted    Accept%      M     f1=0   Target?
--------------------------------------------------------------------------------
     2          418          318      76.1%   1.31        0       yes
     3          267           80      30.0%   2.90        0       yes
     4          222           91      41.0%   2.40        0       yes
     5          173          130      75.1%   1.37        0       yes

====================================================================================================
```

The first draw has 1500 households (1.5 times the request). `M` in these lines is the constant $c$
from the main README, and `f1=0` counts feature cells that hold sample households of that size but
no synthetic ones. The first draw also fixes a quota for each household size, scaled to 1000
households. Six more blocks with the same two tables follow, one for each top-up batch
(`[DIAG-REJECT] Top-up batch 1` to `Top-up batch 6`). Each batch has
`max(ceil(1.7 * shortfall), 200)` households and is filtered against its own proposal
distribution, and accepted households beyond a size's quota are discarded. The run stops when all
quotas are filled, at exactly 1000 households. The acceptance rates are specific to this random
data.

## 3. Inspect the result

```bash
python - <<'EOF'
import numpy as np

r = np.load("outputs/toy_result.npz", allow_pickle=True)
for key in ["pi_g", "pi_m", "household_data", "individual_data", "individual_group_id",
            "household_class", "individual_class", "likelihood"]:
    print(f"{key:20s} {r[key].shape}")
print("converged          ", bool(r["converged"][0]))
print("EM iterations      ", r["likelihood"].size)
print("final log-lik      ", round(float(r["likelihood"][-1]), 3))
EOF
```

Output of the run above:

```text
pi_g                 (4,)
pi_m                 (4, 6)
household_data       (1000, 4)
individual_data      (2539, 5)
individual_group_id  (2539,)
household_class      (1000,)
individual_class     (2539,)
likelihood           (369,)
converged           True
EM iterations       369
final log-lik       -10712.648
```

EM converged after 369 iterations in the restart that was kept. The final population has exactly
1000 households, and the number of individuals follows from the generated household sizes. All
codes and class labels are 1-based. `allow_pickle=True` is needed because `log_phi_g` and
`log_phi_m` are stored as object arrays. Use it only on files you created yourself.

## 4. Choose G and M by BIC

```python
import numpy as np

from hm_popsyn import (
    generate_synthetic_population,
    load_csv,
    prepare_from_merged_table,
    select_model_grid,
)

data = prepare_from_merged_table(load_csv("examples/toy_merged.csv"))
best, best_summary, grid = select_model_grid(
    data.grpdata, data.indgid, data.inddata,
    g_values=[2, 4], m_values=[3, 6], n_restarts=1, max_iter=300, tol=1e-6, seed=0,
)
for s in grid:
    print(f"G={s.G} M={s.M} logL={s.best_log_likelihood:.1f} BIC={s.bic:.1f}")
print("selected:", best_summary.G, best_summary.M)

# Draw from the selected model directly. EMResult stores log probabilities, and this path
# applies no rejection step.
synth = generate_synthetic_population(
    n_households=200,
    pi_g=best.pi_g,
    pi_m=best.pi_m,
    phi_g=[np.exp(x) for x in best.log_phi_g],
    phi_m=[np.exp(x) for x in best.log_phi_m],
    household_size_col=3,
    rng=np.random.default_rng(0),
)
print(synth.household_data.shape, synth.individual_data.shape)
```

```text
G=2 M=3 logL=-10793.0 BIC=22087.2
G=2 M=6 logL=-10756.1 BIC=22378.5
G=4 M=3 logL=-10768.4 BIC=22224.3
G=4 M=6 logL=-10711.4 BIC=22518.2
selected: 2 3
(200, 4) (505, 5)
```

Each grid cell runs `fit_em_with_restarts`. On these random codes the smallest model has the lowest
BIC. To generate with the rejection step, refit with the chosen `G` and `M` through `hm-popsyn-run`
or `fit_and_generate`. To draw from `best` without rejection, pass `best.pi_g`, `best.pi_m` and
`np.exp` of `best.log_phi_g` and `best.log_phi_m` to `generate_synthetic_population`, as in the
last lines above. Passing the log arrays unchanged raises no error and gives wrong probabilities.
