"""
PubExtractor: Data extraction for systematic reviews and meta-analyses.

Reads full text with page anchors, extracts coded values through rules,
semantic scores, and language models, and checks per paper where its text
may be processed.
"""

__version__ = "0.1.3"
__author__ = "Mikyung Shin"
__license__ = "MIT"

from .cells import header_n, parse_cell
from .codebook import import_extraction_schema, load_codebook
from .codebook_table import read_codebook_table, write_codebook_table
from .comparisons import (
    build_comparisons,
    suggest_roles,
    text_arm_n,
    text_statistics,
    text_total_n,
)
from .confidence import (
    confidence_report,
    interpret_confidence,
    needs_escalation,
    score_answer,
    score_extractions,
)
from .correlations import (
    CORRELATION_COLUMNS,
    correlation_checks,
    correlation_matrix,
    correlation_records,
    map_constructs,
    suggest_constructs,
    suggest_mapping,
    to_matrices,
)
from .designs import (
    DESIGN_TYPES,
    method_text,
    suggest_assignment,
    suggest_design,
    suggest_goal,
    suggest_scd_design,
)
from .docx_reader import read_docx, read_docx_images
from .effectsize import (
    EFFECT_COLUMNS,
    METAFOR_COLUMNS,
    METHODS,
    allocate_n,
    cluster_terms,
    compute_effect,
    design_effect,
    effect_rows,
    effect_sizes,
    hedges_j,
    pooled_sd,
    sd_from_mse,
    sd_from_se,
    to_metafor,
)
from .evaluate import (
    AGREEMENT_COLUMNS,
    evaluate,
    graph_agreement,
    icc,
    kappa,
    read_coding_sheet,
    write_coding_sheet,
)
from .evidence import find_keyword_spans, format_evidence, highlight_markdown, search_document
from .figures import (
    ERROR_BARS,
    FIGURE_TYPES,
    figure_type,
    read_bars,
    read_flow,
    read_forest,
    read_scatter,
)
from .fulltext import (
    detect_page_labels,
    detect_sections,
    extract_fulltext_evidence,
    format_anchor,
    read_pdf,
    similarity_scores,
)
from .graphs import (
    DESIGNS,
    OUTCOME_TYPES,
    SCD_COLUMNS,
    calibrate,
    design_rows,
    figure_images,
    find_criterion_lines,
    find_phase_lines,
    find_points,
    find_series,
    graph_rows,
    read_graph,
    split_panels,
    suggest_outcome_type,
    to_scdbayes,
)
from .highlight import highlight_evidence, pattern_from_terms
from .image_tables import (
    find_image_tables,
    manual_table_records,
    read_table_image,
    table_image,
)
from .layers import extract_values
from .licensing import check_llm_use, is_local_endpoint
from .output import to_long, to_wide, write_csv
from .provenance import ProvenanceTracker, compare_provenances, load_provenance, log_human
from .providers import (
    MODEL_SIZES,
    PROVIDERS,
    VISION_MODELS,
    load_env,
    make_responder,
    ollama_models_dir,
    storage_check,
)
from .rag import build_index, chunk_pages, chunk_text, extract_with_rag, retrieve
from .statistics import find_statistics
from .table_roles import GROUP_ROLES, STAT_ROLES, find_groups, map_headers
from .tables import (
    extract_table_values,
    read_document,
    read_header,
    read_pdf_tables,
    read_tables,
    table_grid,
    table_pages,
)
from .trace import apply_decisions, prisma_items, read_decisions, trace_graph, write_trace

__all__ = [
    'read_pdf', 'detect_page_labels', 'detect_sections', 'similarity_scores',
    'extract_fulltext_evidence', 'format_anchor',
    'find_keyword_spans', 'search_document', 'highlight_markdown', 'format_evidence',
    'score_answer', 'interpret_confidence', 'needs_escalation',
    'score_extractions', 'confidence_report',
    'chunk_text', 'chunk_pages', 'build_index', 'retrieve', 'extract_with_rag',
    'check_llm_use', 'is_local_endpoint',
    'pattern_from_terms', 'highlight_evidence',
    'ProvenanceTracker', 'load_provenance', 'compare_provenances', 'log_human',
    'trace_graph', 'write_trace', 'read_decisions', 'apply_decisions', 'prisma_items',
    'PROVIDERS', 'load_env', 'make_responder',
    'MODEL_SIZES', 'VISION_MODELS', 'ollama_models_dir', 'storage_check',
    'read_docx', 'read_docx_images', 'read_pdf_tables',
    'load_codebook', 'import_extraction_schema', 'read_codebook_table', 'write_codebook_table',
    'find_statistics',
    'extract_values', 'to_long', 'to_wide', 'write_csv',
    'read_tables', 'read_document', 'table_grid', 'table_pages', 'read_header', 'parse_cell', 'header_n',
    'STAT_ROLES', 'GROUP_ROLES', 'map_headers', 'find_groups', 'extract_table_values',
    'text_statistics', 'text_arm_n', 'text_total_n', 'build_comparisons', 'suggest_roles', 'METHODS', 'compute_effect', 'effect_sizes',
    'EFFECT_COLUMNS', 'METAFOR_COLUMNS', 'effect_rows', 'to_metafor', 'hedges_j', 'pooled_sd',
    'design_effect', 'cluster_terms', 'allocate_n', 'sd_from_se', 'sd_from_mse',
    'find_image_tables', 'table_image', 'manual_table_records', 'read_table_image',
    'correlation_matrix', 'correlation_checks', 'correlation_records', 'CORRELATION_COLUMNS',
    'map_constructs', 'to_matrices',
    'figure_images', 'calibrate', 'find_points', 'find_series', 'find_phase_lines',
    'find_criterion_lines', 'split_panels', 'read_graph', 'graph_rows', 'design_rows', 'to_scdbayes', 'SCD_COLUMNS', 'DESIGNS', 'OUTCOME_TYPES',
    'figure_type', 'read_bars', 'read_forest', 'read_scatter', 'read_flow', 'FIGURE_TYPES',
    'ERROR_BARS', 'DESIGN_TYPES', 'suggest_design', 'suggest_assignment', 'suggest_scd_design', 'suggest_goal',
    'suggest_outcome_type', 'suggest_constructs', 'suggest_mapping', 'method_text',
    'evaluate', 'graph_agreement', 'kappa', 'icc', 'AGREEMENT_COLUMNS',
    'write_coding_sheet', 'read_coding_sheet',
]


def __getattr__(name):
    if name == 'run_app':
        from .app import run_app
        return run_app
    raise AttributeError(f"module 'pubextractor' has no attribute {name!r}")
