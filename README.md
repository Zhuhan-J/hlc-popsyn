# hlc-popsyn

[![CI](https://github.com/Zhuhan-J/hlc-popsyn/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhuhan-J/hlc-popsyn/actions/workflows/ci.yml)

I ported the hierarchical mixture (hierarchical latent class, HLC) model for population synthesis
by Sun, Erath and Cai (2018) from the authors' MATLAB code to Python. The package fits household
and individual latent classes to a sample of households and their members, generates a synthetic
population, and filters it with a rejection step that corrects relationships between members of
the same household. It is an independent port, and the original authors do not maintain it.

## The original method

The method is described in Lijun Sun, Alexander Erath and Ming Cai, "A hierarchical mixture
modeling framework for population synthesis", Transportation Research Part B 114 (2018), 199-212,
[doi:10.1016/j.trb.2018.06.002](https://doi.org/10.1016/j.trb.2018.06.002). Lijun Sun's MATLAB
code is at
[mcgill-smart-transport/population_synthesis_hierarchical](https://github.com/mcgill-smart-transport/population_synthesis_hierarchical)
(MIT License).

All attributes are categorical. Each household belongs to one of $G$ household classes, and given
its class, each household attribute (household size included) follows its own categorical
distribution. Each member belongs to one of $M$ individual classes, with weights that depend on
the household class. The individual classes are universal: all household classes share their
attribute distributions. For household $i$ with household attributes $x_{ik}$, $k = 1, \dots, h$,
and members $j = 1, \dots, n_i$ with attributes $x_{ijk}$, $k = h+1, \dots, K$,

```math
p(\mathbf{x}_i) = \sum_{g=1}^{G} \lambda_g \prod_{k=1}^{h} \phi^{(k)}_{x_{ik} g} \prod_{j=1}^{n_i} \sum_{m=1}^{M} \mu_{gm} \prod_{k=h+1}^{K} \theta^{(k)}_{x_{ijk} m}
```

In the code, `pi_g` and `pi_m` hold $\lambda$ and $\mu$, and `log_phi_g` and `log_phi_m` hold
$\log\phi$ and $\log\theta$. The parameters are estimated by EM. For each $(G, M)$ the paper runs EM
20 times from random starts and keeps the run with the highest log-likelihood, and it chooses $G$
and $M$ by a grid search on $\mathrm{BIC} = -2\log L + D_f \log \sum_i n_i$, where
$L = \prod_i p(\mathbf{x}_i)$, $d_k$ is the number of categories of attribute $k$, and

```math
D_f = (G-1) + G(M-1) + G\sum_{k=1}^{h}(d_k - 1) + M\sum_{k=h+1}^{K}(d_k - 1)
```

A synthetic household is drawn by sampling its class and household attributes, then as many
members as its household-size attribute gives, each with an individual class and attributes.
Members are independent given the household class, so pairs such as couples are not reproduced
well. For two-person households the paper adds a rejection step. Let $y_1$ be the absolute age
difference and $y_2$ the absolute sex difference of the two members, $f_1$ their distribution in
the synthetic population and $f_2$ their distribution in the sample. A household is accepted with
probability

```math
\alpha_i = \frac{f_2(y_{i1}, y_{i2})}{c \, f_1(y_{i1}, y_{i2})}, \qquad c = \max_{y_1, y_2} \frac{f_2(y_1, y_2)}{f_1(y_1, y_2)}
```

The paper names this constant $M$. I write $c$ to keep $M$ for the number of individual classes,
but the diagnostics printed by the code still call it `M`. The paper adds that "the same procedure
can be applied for other household types (e.g., three-person)".

## What I implemented and changed

I translated the universal-class EM (`EM_multilevel_across.m`), the parameter count, BIC and
restarts (`run_multilevel_1.m`), the generator (`run_create_2.m`), the rejection step
(`run_two_within_the_same_4.m`), Cramér's V and the MATLAB and MEX helpers into a Python package
with a command-line tool, `hm-popsyn-run`. Category codes stay 1-based, as in MATLAB. On top of
the port I changed the rejection step:

1. Rejection covers every household size of two or more, with separate target and proposal
   distributions for each size. A household's two features are the rounded mean absolute
   differences between its first listed member and each other member, in ordinal age positions
   and in coarse employment groups (in place of sex). For two-person households the mean is the
   paper's pairwise difference. The rounded age difference is then grouped into 5 bins.
2. I compute $c$ only over cells where both distributions are positive, cap it at 3 (close to the
   value of 2.97 reported in the paper) and clip the acceptance probability at 1, so cells where
   $f_2/f_1$ exceeds 3 stay below their target share. A household size with fewer than 30 sample
   households with valid features skips the test, and all its households are kept.
3. `generate_with_multi_size_rejection` returns exactly `n_households` households. It draws 1.5
   times that number and fixes a quota for each household size from this first draw, scaled to
   `n_households`. Accepted households fill the quotas and any beyond a quota are discarded. While
   a quota is short, it draws a fresh batch of `max(ceil(1.7 * shortfall), 200)` households and
   filters it against that batch's own proposal distribution, and it raises `RuntimeError` after
   10 batches. The output keeps the size mix of the first draw, and no household appears twice.

Two parts differ from the MATLAB scripts but follow the paper (Sec. 4.2): `fit_em_with_restarts`
keeps the restart with the highest log-likelihood, which for fixed $G$ and $M$ is the lowest BIC,
and `select_model_grid` runs the $(G, M)$ grid search. I also adapted `Gibbs_universal.m`, which
the paper does not describe, as `gibbs_universal`, with symmetric Dirichlet priors in place of the
MATLAB stick-breaking priors on the class weights. The pipeline does not use it.

Extensions I use in my own research are not part of this package.

### Model architecture and data flow

```mermaid
flowchart TD
    csv[("merged CSV")] --> load["io.load_csv"]
    load --> prep["io.prepare_from_merged_table"]
    prep -->|"grpdata, indgid, inddata"| fit["em.fit_em_with_restarts"]
    fit -->|"calls once per restart"| core["em.em_multilevel_across_universal"]
    fit -->|"restart with lowest BIC"| params["EMResult"]
    params --> gen["synthesis_eval.generate_synthetic_population"]
    gen --> raw["synthetic_raw"]
    params --> draw
    subgraph rejloop ["synthesis_eval.generate_with_multi_size_rejection"]
        draw["generate_synthetic_population, first draw then top-up batches"] --> rej["apply_rejection_by_size"]
        rej --> feat["extract_pair_features_by_size"]
        feat --> filt["rejection_filter_by_size, one call per household size"]
        filt -->|"top up while short"| draw
    end
    prep -->|"input sample as target"| feat
    filt -->|"exactly n_households"| final["synthetic_final"]
    final --> npz[("result .npz")]
    params -->|"parameters, likelihood trace, converged flag"| npz
    prep -.-> grid["em.select_model_grid"]
    grid -.->|"once per G, M pair"| fit
    prep -.-> gibbs["gibbs.gibbs_universal"]
    gibbs -.->|"posterior means"| gen
    raw -.-> ev["evaluation.cramer_v_from_counts, evaluation.marginal_distribution"]
    final -.-> ev
```

`hm-popsyn-run` (`runner.main`) follows the solid arrows. It loads and prepares the CSV, calls
`pipeline.fit_and_generate` for every step from `fit_em_with_restarts` to `synthetic_final`, and
writes the fitted parameters, the EM likelihood trace, the convergence flag and `synthetic_final`
to an `.npz` file. `synthetic_raw` is a separate draw without rejection that `fit_and_generate`
returns for comparison. Dashed arrows are functions you call yourself. EM uses `kernels.sum_eik`
and `kernels.logsumexp`.

## Comparison with the MATLAB code

| Component | Original MATLAB | This package | Status |
| --- | --- | --- | --- |
| Model with universal individual classes | `EM_multilevel_across.m` | `em.em_multilevel_across_universal` | Same |
| Model with class-specific individual classes | `EM_multilevel.m` | Not included | Not ported |
| EM start | Normalized Gamma(1, `con`) draws, an optional tensor-factorization start, or user-supplied values (`initial`) | Normalized Gamma(1, `concentration`) draws or user-supplied values (`init`); no tensor start. The scale cancels after normalization in both versions, so `--concentration` has no effect on EM | Different |
| E-step, M-step, log-likelihood | `EM_multilevel_across.m` | Same updates in the same order. Adds 1e-12 where MATLAB adds machine epsilon, and also floors denominators and log arguments at 1e-12 | Same |
| Stopping rule | Relative change of the log-likelihood at most `tol`, checked from iteration 3; at most `maxIter - 1` iterations | Same rule, checked from iteration 2; at most `max_iter` iterations; adds a `converged` flag | Same |
| Parameter count and BIC | `run_multilevel_1.m`: best log-likelihood of the run, log of the number of individuals | `em.count_parameters_universal`, `em.bic_score` | Same |
| Restarts | 20 runs saved; `run_create_2.m` uses a run picked by hand | `em.fit_em_with_restarts` keeps the lowest BIC (highest log-likelihood), as in the paper, Sec. 4.2 | Different |
| G and M search | Loads a precomputed `bic.mat`; the grid code is not in the repository | `em.select_model_grid`, the grid search described in the paper, Sec. 4.2 | Added |
| Gibbs sampler | Stick-breaking priors on class weights with resampled concentration parameters; returns every draw | Symmetric Dirichlet priors on all parameters; returns posterior means after burn-in and thinning, and class-label traces | Different |
| Input preparation | Sorts rows, shifts each column to a minimum of 1, computes household size from row counts and inserts it as the fourth household attribute | Same sort and shift; household size must already be a column and is shifted with the others; household ids are remapped to 1..N | Different |
| Generation | Household class, household attributes, members from the size attribute; member attributes from the class mixture; 10 times the number of sample households | Same distribution; draws and returns each member's class; generates `n_households` | Same |
| Rejection scope | Two-person households, one pass, sample as target | Every size of two or more, one target and proposal per size, sample as target; sizes with fewer than 30 valid sample households are skipped | Different |
| Rejection features | Absolute differences of the raw age and sex codes, plus 1 | Rounded mean absolute differences to the first member, in ordinal age positions (then 5 bins) and in employment groups | Different |
| Constant $c$ | Maximum over all cells | Maximum over cells where both distributions are positive, capped at 3; acceptance probability clipped at 1 | Different |
| Output households | Rejects from 8 stacked copies of the synthetic rows; count not controlled | Exactly `n_households`: per-size quotas from the first draw, filled by fresh batches that are each filtered against their own proposal distribution | Different |
| Evaluation | Cramér's V with smoothing 1e-6; marginals with `sumdims` | Same Cramér's V formula with smoothing 1e-12 (`cramer_v_from_counts`); `marginal_distribution` | Same |
| Figures | `run_draw_figures_3.m` and plots in the other scripts | Not included | Not ported |
| Helpers | `int_hist.c`, `sum_eik.c`, `mnrnd_new.m`, `logsumexp.m`, `sumdims.m`, `randgamma.c` | `kernels.py` ports all but `randgamma`, which no MATLAB script calls; `sum_eik` adds a range check and `mnrnd_new` normalizes rows | Same |

## Install, run and reproduce

### Install

```bash
git clone https://github.com/Zhuhan-J/hlc-popsyn.git
cd hlc-popsyn
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package needs Python 3.10 or newer and NumPy 1.24 or newer. CI runs the lint and tests on
Python 3.10, 3.11 and 3.12.

### Tests

```bash
ruff check .
pytest -q
```

```text
All checks passed!
...........                                                              [100%]
11 passed in 1.42s
```

The tests use small arrays defined inline. They check shapes, normalization and label ranges for
the helpers, EM, the Gibbs sampler and the generator, and run the pipeline end to end. Their
inputs fall under the 30-household guard, so they do not test acceptance rates. No test compares
results with MATLAB output; the comparison table comes from reading the two code bases side by
side.

### Toy example

```bash
python examples/make_toy_data.py
hm-popsyn-run --input examples/toy_merged.csv --output outputs/toy_result.npz \
    --G 4 --M 6 --n-households 1000 --n-restarts 2 --max-iter 500 --tol 1e-6 --seed 1
```

These commands rewrite the committed toy table and write a synthetic population of 1000 households
to `outputs/toy_result.npz`; [examples/README.md](examples/README.md) shows the output and how to
read the result file.

`hm-popsyn-run --help` lists all options. The command defaults to 5 restarts, 1000 EM iterations,
`--tol 1e-8` and `--seed 12345`. Called directly, `fit_em_with_restarts` defaults to the settings
of the paper and `run_multilevel_1.m`: 20 restarts, a tolerance of 1e-10 and at most 4000
iterations. The help text of `--no-rejection` still says "two-person", but the rejection step
covers all household sizes of two or more.

### Input format

`hm-popsyn-run` reads a numeric CSV with one header line and one row per individual, laid out like
the MATLAB table after its household-size column is inserted. With the default column mapping of
`prepare_from_merged_table` (0-based):

| Columns | Content |
| --- | --- |
| 0 | household id |
| 1 | individual id within the household |
| 2, 3, 4, 5 | household attributes; column 5 is the household size |
| 6, 7, 8, 10, 11 | individual attributes |
| 9 | not used |

All values must be integer codes. Each column is shifted so that its minimum becomes 1, and the
generator uses the household-size code directly as the number of members. The sample therefore
needs at least one single-person household, and each size code should equal the number of rows of
its household.

The rejection step reads individual attribute 0 as an ordinal age code and individual attribute 1
as the attribute it maps to groups. The default lookup tables (`AGE_CODE_TO_ORDER` and
`EMPLOY_CODE_TO_GROUP` in `synthesis_eval.py`) assume one specific coding. For other codings, pass
your own tables to `fit_and_generate` through `age_code_to_order` and `employ_code_to_group`; the
command-line tool always uses the defaults. The age binning assumes 18 ordinal positions
(`AGE_DIFF_MAX = 17`) in both cases. Members with codes outside the tables are ignored, and a household
with no valid pair that includes its first member skips the test and is kept.

### Python API

```python
from hm_popsyn import fit_and_generate, load_csv, marginal_distribution, prepare_from_merged_table

data = prepare_from_merged_table(load_csv("examples/toy_merged.csv"))
result = fit_and_generate(
    data.grpdata, data.indgid, data.inddata,
    G=4, M=6, n_households=1000, n_restarts=2, max_iter=500, tol=1e-6, seed=1,
)
synth = result.synthetic_final
print(synth.household_data.shape, synth.individual_data.shape)
# Household size is household attribute 3 in the toy data.
print("input    ", marginal_distribution(data.grpdata[:, 3]).round(3))
print("synthetic", marginal_distribution(synth.household_data[:, 3]).round(3))
```

After the rejection diagnostics, this prints:

```text
(1000, 4) (2539, 5)
input     [0.272 0.274 0.184 0.148 0.122]
synthetic [0.28  0.279 0.178 0.148 0.115]
```

The toy codes are independent random numbers, so these numbers show only that the code runs.

## Attribution and license

The method is by Sun, Erath and Cai; the MATLAB code, linked above, is by Lijun Sun and released
under the MIT License. Its README notes that some functions come from the MATLAB Tensor Toolbox
and the [Lightspeed](https://github.com/tminka/lightspeed) toolbox; `logsumexp.m`, which I
ported, credits Tom Minka. This repository contains no survey or other real data:
`examples/toy_merged.csv` holds random integer codes written by `examples/make_toy_data.py`.

The package is released under the MIT License ([LICENSE](LICENSE)). The license keeps the original
copyright notice (2017, Lijun Sun) and adds mine (2026, Zhuhan Jin).

## Citation

If you use this package, please cite the paper:

```bibtex
@article{sun2018hierarchical,
  title={A hierarchical mixture modeling framework for population synthesis},
  author={Sun, Lijun and Erath, Alexander and Cai, Ming},
  journal={Transportation Research Part B: Methodological},
  volume={114},
  pages={199--212},
  year={2018},
  publisher={Elsevier}
}
```

Citation metadata for the software itself is in [CITATION.cff](CITATION.cff).
