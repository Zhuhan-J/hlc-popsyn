# Toy example

This example runs the full workflow on a small table of random integer codes. The table has no
connection to any real data; it only has the column layout that `hm-popsyn-run` expects (see
"Input format" in the main README). Run the commands from the repository root after
`pip install -e ".[dev]"`.

## 1. Make the toy data

```bash
python examples/make_toy_data.py
```

```text
Wrote 1287 individuals in 500 households to examples/toy_merged.csv
```

The generated file is committed as `examples/toy_merged.csv` (about 35 KB). The script uses a
fixed seed (`--seed 2018`) and 500 households by default, so rerunning it reproduces the same
file. Household sizes are drawn from 1 to 5, and all other attributes are uniform random codes.

## 2. Fit the model and generate a synthetic population

```bash
hm-popsyn-run --input examples/toy_merged.csv --output outputs/toy_result.npz \
    --G 4 --M 6 --n-households 1000 --n-restarts 2 --max-iter 500 --tol 1e-6 --seed 1
```

The command fits EM with 2 restarts, generates 1000 households with the rejection step, and
writes `outputs/toy_result.npz`. With these settings it finished in under a minute when this
README was written. It prints only the rejection diagnostics. The first block of the real output
was:

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

Six more blocks with the same two tables follow, one for each top-up batch
(`[DIAG-REJECT] Top-up batch 1` to `Top-up batch 6`), until exactly 1000 households have been
accepted. The acceptance rates are specific to this random data.

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
1000 households; the number of individuals follows from the generated household sizes.
`allow_pickle=True` is needed because the per-attribute parameter arrays `log_phi_g` and
`log_phi_m` are stored as object arrays. Only load files you created yourself this way.
