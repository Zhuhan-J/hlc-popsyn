"""Hierarchical mixture population synthesis package."""

from .em import (
    EMResult,
    ModelSelectionResult,
    bic_score,
    count_parameters_universal,
    em_multilevel_across_universal,
    fit_em_with_restarts,
    select_model_grid,
)
from .evaluation import cramer_v_from_counts, marginal_distribution
from .gibbs import GibbsResult, gibbs_universal
from .io import PreparedData, load_csv, prepare_from_merged_table
from .kernels import int_hist, logsumexp, mnrnd_new, sum_eik, sumdims
from .metrics import (
    category_sets_from,
    evaluate_synthetic,
    member_table,
    precision_recall_f1,
    srmse,
)
from .pipeline import PipelineResult, fit_and_generate
from .synthesis_eval import (
    SynthesisResult,
    apply_rejection_by_size,
    apply_two_person_rejection,
    extract_pair_features_by_size,
    generate_synthetic_population,
    generate_with_multi_size_rejection,
    rejection_filter_by_size,
    rejection_filter_two_person_households,
    two_person_pair_features,
)

__all__ = [
    "EMResult",
    "ModelSelectionResult",
    "bic_score",
    "count_parameters_universal",
    "em_multilevel_across_universal",
    "fit_em_with_restarts",
    "select_model_grid",
    "cramer_v_from_counts",
    "marginal_distribution",
    "GibbsResult",
    "gibbs_universal",
    "PreparedData",
    "load_csv",
    "prepare_from_merged_table",
    "category_sets_from",
    "evaluate_synthetic",
    "member_table",
    "precision_recall_f1",
    "srmse",
    "int_hist",
    "logsumexp",
    "mnrnd_new",
    "sum_eik",
    "sumdims",
    "PipelineResult",
    "fit_and_generate",
    "SynthesisResult",
    "apply_rejection_by_size",
    "apply_two_person_rejection",
    "extract_pair_features_by_size",
    "generate_synthetic_population",
    "generate_with_multi_size_rejection",
    "rejection_filter_two_person_households",
    "rejection_filter_by_size",
    "two_person_pair_features",
]
