# API Reference

## Reading documents

| Name | Signature | Purpose |
|---|---|---|
| `read_pdf` | `read_pdf(path, max_pages=None)` | Read a born-digital PDF into per-page text. |
| `detect_page_labels` | `detect_page_labels(pages)` | Attach the page number printed in the document to each page. |
| `detect_sections` | `detect_sections(pages)` | Label each page with section headings and where they sit in its text. |
| `read_docx` | `read_docx(path)` | Blocks of a DOCX file in body order, each with its section and number. |
| `read_docx_images` | `read_docx_images(path)` | Images embedded in a DOCX file, numbered in the order they appear. |
| `read_pdf_tables` | `read_pdf_tables(path, max_pages=None)` | Tables in a PDF, numbered across the document, with file and printed page. |
| `read_tables` | `read_tables(path, max_pages=None)` | Cell grids for every table in a PDF, DOCX, or image file, with caption and note. |
| `read_document` | `read_document(path, max_pages=None)` | Pages and table grids from one pass over a PDF, DOCX, or image file. |
| `table_grid` | `table_grid(table)` | One cell grid for a ``read_pdf_tables`` table or a ``read_docx`` table block. |
| `table_pages` | `table_pages(grids)` | Tables as page-like blocks, so ``extract_values`` and ``extract_with_rag`` search them. |
| `read_header` | `read_header(grid, header_rows=None)` | Header rows above the numeric body, joined into one label per column. |
| `parse_cell` | `parse_cell(text)` | Parse a table cell into a kind (number, percent, mean_sd, n_percent, missing, empty, text) and its values. |
| `header_n` | `header_n(text)` | Split a header label from the sample size it states, such as `Treatment (n = 24)`. |

## Codebook and extraction

| Name | Signature | Purpose |
|---|---|---|
| `load_codebook` | `load_codebook(source)` | Fields from a dict or a YAML file, checked and normalized. |
| `import_extraction_schema` | `import_extraction_schema(source)` | Codebook from the ``extraction_schema.yml`` layout. |
| `read_codebook_table` | `read_codebook_table(path)` | Codebook from a long .xlsx or .csv sheet, one row per term. |
| `write_codebook_table` | `write_codebook_table(codebook, path)` | Write a codebook as a long .xlsx or .csv sheet that ``read_codebook_table`` reads back. |
| `find_statistics` | `find_statistics(text)` | Find reported n, mean, SD, SE, t, F, r, p, and MSE values in text, with their positions. |
| `extract_values` | `extract_values(pages, codebook, paper_id=None, respond=None, encoder=None, permission=None, endpoint=None, tracker=None)` | One record per extracted value, rules first. |
| `to_long` | `to_long(records)` | One row per extracted value, columns in ``COLUMNS`` order. |
| `to_wide` | `to_wide(records)` | One row per paper, one column per field; values joined with '; '. |
| `write_csv` | `write_csv(rows, path)` | Write rows to a UTF-8 CSV with a header from every key seen. |

## Evidence and search

| Name | Signature | Purpose |
|---|---|---|
| `extract_fulltext_evidence` | `extract_fulltext_evidence(path, inclusion_patterns, model=None, max_pages=None, window_words=0)` | Locate criterion evidence in a PDF, anchored to page and section. |
| `similarity_scores` | `similarity_scores(texts, description, model)` | Cosine similarity of each text to a description, on the encoder's embeddings. |
| `format_anchor` | `format_anchor(span)` | Citation anchor for a span, as the reviewer sees it. |
| `find_keyword_spans` | `find_keyword_spans(text, keyword, window=5, case_sensitive=False, max_matches=None)` | Find a keyword in text with surrounding context. |
| `search_document` | `search_document(pages, keywords, window=5, case_sensitive=False, max_per_page=None)` | Search a parsed document for keywords, anchored to printed pages. |
| `highlight_markdown` | `highlight_markdown(span)` | Render a span with the matched term in markdown bold, for notebooks and sheets. |
| `format_evidence` | `format_evidence(spans, separator=' \| ', max_spans=3, with_anchor=True)` | Join spans into one evidence string for a report column. |
| `pattern_from_terms` | `pattern_from_terms(terms)` | Build a search pattern from plain terms, so nobody has to write a regex. |
| `highlight_evidence` | `highlight_evidence(xlsx_path, patterns, fields=None, colours=None, default_colour='FFFF0000', key_column=None)` | Color the matched terms inside a workbook, in place. |

## Retrieval and confidence

| Name | Signature | Purpose |
|---|---|---|
| `chunk_text` | `chunk_text(text, chunk_size=1000, overlap=200)` | Split text into overlapping character chunks, breaking at whitespace. |
| `chunk_pages` | `chunk_pages(pages, chunk_size=1000, overlap=200)` | Chunk each page separately, so every chunk keeps the page it came from. |
| `build_index` | `build_index(chunks, model, batch_size=32)` | Embed chunks once for reuse across queries. |
| `retrieve` | `retrieve(query, chunks, embeddings, model, top_k=5)` | Rank chunks against a query by cosine similarity. |
| `extract_with_rag` | `extract_with_rag(text, queries, respond, model, top_k=5, chunk_size=1000, overlap=200, template=None, tracker=None, threshold=0.7, permission=None, endpoint=None, paper_id=None, cache=None)` | Answer questions about a document from retrieved passages. |
| `score_answer` | `score_answer(answer, passages=())` | Score how well an extracted answer is grounded in its source passages. |
| `interpret_confidence` | `interpret_confidence(score)` | Band label for a grounding score. |
| `needs_escalation` | `needs_escalation(score, threshold=0.7)` | Whether an extraction should go to a second reviewer. |
| `score_extractions` | `score_extractions(extractions, passages_list=None, answer_key='answer', threshold=0.7)` | Score a list of extractions and flag the ones needing a second reviewer. |
| `confidence_report` | `confidence_report(scored, threshold=0.7)` | Counts by band and how many extractions need a second reviewer. |

## License and providers

| Name | Signature | Purpose |
|---|---|---|
| `check_llm_use` | `check_llm_use(content='fulltext', access='subscription', license_type=None, ai_terms='unknown', processing='local', region='us', provider_terms='standard', research_org=True)` | Decide whether one paper may be sent to a language model. |
| `is_local_endpoint` | `is_local_endpoint(url)` | True when a model endpoint names a host on this machine. |
| `PROVIDERS` | `PROVIDERS` | Base URL and key variable of each supported chat API. |
| `load_env` | `load_env(path='.env', environ=None)` | Read KEY=VALUE lines into the environment without overriding variables already set. |
| `make_responder` | `make_responder(provider, model, base_url=None, key_env=None, environ=None, timeout=120, post=None)` | Build ``respond(prompt, images=None)`` for an OpenAI-compatible chat API. |
| `MODEL_SIZES` | `MODEL_SIZES` | Approximate download size in GB of local models, as listed in the Ollama library. |
| `VISION_MODELS` | `VISION_MODELS` | Local models that read images. |
| `ollama_models_dir` | `ollama_models_dir(environ=None)` | Folder where Ollama keeps models: OLLAMA_MODELS, else ``~/.ollama/models``. |
| `storage_check` | `storage_check(model='gemma3:4b', path=None, environ=None, base_url='http://localhost:11434')` | Free disk space against the download size of a local model. |

## Provenance

| Name | Signature | Purpose |
|---|---|---|
| `ProvenanceTracker` | `ProvenanceTracker(stage, task)` | Record what a run did, so a reported result can be reconstructed. |
| `ProvenanceTracker.log_permission` | `log_permission(permission, paper_id=None)` | Record the license verdict each paper was processed under. |
| `ProvenanceTracker.log_prompt` | `log_prompt(prompt, model=None, parameters=None)` | Record a prompt by SHA-256 hash, with its model and a running digest. |
| `ProvenanceTracker.log_retrieval` | `log_retrieval(embedding_model=None, chunk_size=None, chunk_overlap=None, top_k=None, **kwargs)` | Record retrieval settings. |
| `ProvenanceTracker.log_model_config` | `log_model_config(model_name, model_type=None, checkpoint=None, **kwargs)` | Record the model name, type, and checkpoint. |
| `ProvenanceTracker.log_random_seed` | `log_random_seed(seed)` | Record the random seed. |
| `ProvenanceTracker.log_data` | `log_data(path=None, n_records=None, n_labeled=None, **kwargs)` | Record the data file and its record counts. |
| `ProvenanceTracker.log_threshold` | `log_threshold(threshold, target_recall=None, basis=None)` | Record a decision threshold and its basis. |
| `ProvenanceTracker.log_criteria` | `log_criteria(criteria)` | Record criterion descriptions verbatim. |
| `ProvenanceTracker.add_note` | `add_note(key, value)` | Record a free note under ``key``. |
| `ProvenanceTracker.to_dict` | `to_dict()` | Everything recorded, as a dict. |
| `ProvenanceTracker.save` | `save(path)` | Write the record as JSON. |
| `load_provenance` | `load_provenance(path)` | Read a provenance file written by `ProvenanceTracker`. |
| `compare_provenances` | `compare_provenances(first, second)` | Report which recorded fields differ between two runs. |
| `log_human` | `log_human(tracker, paper_id, field, action, before, after, coder=None, time=None)` | Record one coder step with coder and time (now in UTC when ``time`` is None). |
| `trace_graph` | `trace_graph(records, effects=(), graph_rows=(), tracker=None)` | Per paper: nodes, edges, evidence, and status per field. |
| `write_trace` | `write_trace(graphs, path)` | Write trace graphs as JSON. |
| `read_decisions` | `read_decisions(path)` | Read a coder decisions file (confirm, correct, not_reported). |
| `apply_decisions` | `apply_decisions(records, decisions, tracker=None)` | Apply decisions as layer `human`; return records and the human steps. |
| `prisma_items` | `prisma_items(tracker, records)` | Methods text for PRISMA 2020 items 9 and 11. |

## Table values

| Name | Signature | Purpose |
|---|---|---|
| `STAT_ROLES` | `STAT_ROLES` | Statistic roles a column can take. |
| `GROUP_ROLES` | `GROUP_ROLES` | Group roles an arm can take: treatment, control, other. |
| `map_headers` | `map_headers(header, overrides=None, respond=None, permission=None, endpoint=None, tracker=None, paper_id=None)` | Stat role and time point of each column: rules, then a model, then ``overrides``. |
| `find_groups` | `find_groups(grid, header, overrides=None, respond=None, permission=None, endpoint=None, tracker=None, paper_id=None)` | Arm labels in the rows or the header, each with its group role. |
| `extract_table_values` | `extract_table_values(grids, paper_id=None, header_rows=None, header_overrides=None, group_overrides=None, notes=None, respond=None, permission=None, endpoint=None, tracker=None)` | One record per table value, in the record format of ``extract_values``. |

## Effect sizes

| Name | Signature | Purpose |
|---|---|---|
| `text_statistics` | `text_statistics(text, paper_id=None, outcome=None, time=None, where=None)` | t, F, and MSE stated in ``text`` as records that ``build_comparisons`` can attach. |
| `text_arm_n` | `text_arm_n(text, arms, paper_id=None, where=None)` | n records for arms named just before an 'n = N' in ``text``. |
| `text_total_n` | `text_total_n(text)` | The total N when ``text`` states one value as 'N = ...', else None. |
| `build_comparisons` | `build_comparisons(records, studies=None, pairing=None, inputs=None)` | One comparison per treatment arm, outcome, and time point, paired with its control. |
| `METHODS` | `METHODS` | Effect-size methods in order of preference. |
| `compute_effect` | `compute_effect(comparison, methods=('adjusted_difference', 'posttest_difference', 'from_t', 'from_f', 'difference_in_differences'), allow_did=False)` | Hedges' g and its variance by the first method in ``methods`` the data allow. |
| `effect_sizes` | `effect_sizes(comparisons, methods=('adjusted_difference', 'posttest_difference', 'from_t', 'from_f', 'difference_in_differences'), allow_did=False)` | ``compute_effect`` for each comparison. |
| `EFFECT_COLUMNS` | `EFFECT_COLUMNS` | Column order of `effect_rows`. |
| `METAFOR_COLUMNS` | `METAFOR_COLUMNS` | Column order of `to_metafor`. |
| `effect_rows` | `effect_rows(effects)` | One flat row per effect, columns in ``EFFECT_COLUMNS`` order, for ``write_csv``. |
| `to_metafor` | `to_metafor(effects)` | Rows with ``yi`` and ``vi`` for metafor, one per effect with a variance. |
| `hedges_j` | `hedges_j(df)` | Small-sample correction J = 1 - 3 / (4 df - 1). |
| `pooled_sd` | `pooled_sd(sd_t, sd_c, n_t, n_c)` | Pooled SD of two arms. |
| `design_effect` | `design_effect(n_total, clusters, icc)` | 1 + (m - 1) ICC with m = analyzed N / clusters. |
| `cluster_terms` | `cluster_terms(n_total, clusters, icc)` | Small-number-of-clusters factor gamma and the degrees of freedom for J. |
| `allocate_n` | `allocate_n(n_total, weights)` | Split a total N over arms in proportion to ``weights``, rounding half to even. |
| `sd_from_se` | `sd_from_se(se, n, r2=0.0)` | Unadjusted SD from the SE of a group mean; ``r2`` > 0 for covariate-adjusted SEs. |
| `sd_from_mse` | `sd_from_mse(mse, r2)` | Unadjusted posttest SD from the ANCOVA error mean square; ``r2`` is rho^2 for one pretest. |

## Image tables

| Name | Signature | Purpose |
|---|---|---|
| `find_image_tables` | `find_image_tables(path)` | Tables in a PDF whose body is a picture with no text, numbered as ``read_tables`` numbers them. |
| `table_image` | `table_image(path, pdf_page, bbox, resolution=200)` | PNG bytes of one page region, rendered at ``resolution`` dots per inch. |
| `manual_table_records` | `manual_table_records(rows, paper_id, table, pdf_page=None, printed_page=None, caption=None, layer='human')` | Records for values a coder typed, or confirmed from a vision read (``layer='vision'``), in the ``extract_table_values`` format. |
| `read_table_image` | `read_table_image(png, respond, permission, endpoint, paper_id=None, table=None, pdf_page=None, tracker=None)` | Values a vision model reads from a table image, every one left for a coder to confirm. |

## Study design

| Name | Signature | Purpose |
|---|---|---|
| `DESIGN_TYPES` | `DESIGN_TYPES` | Designs `suggest_design` returns: group, single_case, correlational, qualitative. |
| `suggest_design` | `suggest_design(pages, grids=(), figures=())` | Suggested design of one paper and the evidence, for the coder to confirm. |
| `method_text` | `method_text(pages)` | Text from the first Method heading to Results, or all text before References. |

## Correlation matrices

| Name | Signature | Purpose |
|---|---|---|
| `CORRELATION_COLUMNS` | `CORRELATION_COLUMNS` | Column order of `correlation_records`. |
| `correlation_matrix` | `correlation_matrix(grid, n=None)` | Labels, square matrix, n, and location of a correlation table grid. |
| `correlation_checks` | `correlation_checks(matrix, layout, n)` | Flags for a correlation matrix; nothing is repaired. |
| `correlation_records` | `correlation_records(results, paper_id=None)` | One record per variable pair in ``CORRELATION_COLUMNS`` order, for ``write_csv``. |
| `map_constructs` | `map_constructs(labels, constructs, overrides=None)` | Construct for each label by exact, then normalized, match; ``overrides`` always win. |
| `suggest_roles` | `suggest_roles(records, pages=None)` | A suggested role and its reason for each table arm without a treatment or control group. |
| `suggest_scd_design` | `suggest_scd_design(pages)` | The single-case design named most often in the Method text, and the evidence. |
| `suggest_goal` | `suggest_goal(pages, caption='')` | 'decrease' or 'increase' as the intervention goal of a single-case graph, and the evidence. |
| `suggest_assignment` | `suggest_assignment(pages)` | 'rct' or 'qed' for a group study, and the evidence, from its Method text. |
| `suggest_constructs` | `suggest_constructs(label_sets)` | Construct names suggested from the labels of several studies, most shared first. |
| `suggest_mapping` | `suggest_mapping(labels, constructs, overrides=None, threshold=0.8)` | A construct and its reason for each label, for a coder to confirm or change. |
| `to_matrices` | `to_matrices(studies, constructs, folder)` | Per-study CSV matrices over ``constructs`` and ``n.csv``. |

## Single-case graphs

| Name | Signature | Purpose |
|---|---|---|
| `SCD_COLUMNS` | `SCD_COLUMNS` | Column order of `graph_rows` and `to_scdbayes`. |
| `DESIGNS` | `DESIGNS` | Single-case designs `read_graph` reads: ab, reversal, multiple_baseline, multiple_probe, alternating, changing_criterion. |
| `OUTCOME_TYPES` | `OUTCOME_TYPES` | Outcome types `graph_rows` snaps to: continuous, count, binary, proportion, ordinal. |
| `suggest_outcome_type` | `suggest_outcome_type(caption='', words=())` | Outcome type of a single-case graph from its axis and caption words, and the evidence. |
| `figure_images` | `figure_images(path)` | Figures in a PDF or DOCX as ``figure``, ``pdf_page``, ``caption``, ``png`` bytes, and ``words``. |
| `calibrate` | `calibrate(x_ticks, y_ticks)` | Linear pixel-to-value map per axis from two or more ``(pixel, value)`` ticks. |
| `find_points` | `find_points(png, colour=None, radius=None)` | Pixel centers ``(x, y)`` of filled or open markers inside the axes, ordered by x. |
| `find_series` | `find_series(png, radius=None)` | Markers as ``(x, y, series)``, the series named by shape, color, or both where they vary. |
| `find_phase_lines` | `find_phase_lines(png)` | Pixel x of vertical lines, solid or dashed, inked over half the axes height or more. |
| `find_criterion_lines` | `find_criterion_lines(png)` | Horizontal segments inside the axes, at least 15% of their width, as ``(y, x0, x1)``. |
| `split_panels` | `split_panels(png, count=None)` | Stacked panels split at blank rows, or into ``count`` equal bands. |
| `read_graph` | `read_graph(png, design='ab', count=None)` | Panels of a figure as the design reads them, in pixels of the whole figure. |
| `graph_rows` | `graph_rows(points, calibration, phase_lines=(), outcome_type='continuous', levels=None, criterion_lines=None, area=None, **fields)` | scdbayes rows from marker pixels: values, sessions, phases, and snapped outcomes. |
| `design_rows` | `design_rows(panels, calibration, design='ab', cases=(), **fields)` | Rows for every panel of ``read_graph``, one case per panel. |
| `to_scdbayes` | `to_scdbayes(rows, path)` | Write graph rows to a CSV whose case, session, and phase columns scdbayes detects. |

## Other figures

| Name | Signature | Purpose |
|---|---|---|
| `FIGURE_TYPES` | `FIGURE_TYPES` | Kinds `figure_type` suggests: scd_line, bar, scatter, forest, flow. |
| `ERROR_BARS` | `ERROR_BARS` | Error bar kinds `read_bars` reads: se, sd, ci. |
| `figure_type` | `figure_type(png, caption='', design=None)` | 'scd_line', 'bar', 'scatter', 'forest', or 'flow' as a suggestion; a confirmed ``design`` is the prior. |
| `read_bars` | `read_bars(png, calibration, names=(), error='se', paper_id=None, figure=None, pdf_page=None)` | Mean and error records per bar; a bar with no whisker gives no error value and ``no_error_bars``. |
| `read_forest` | `read_forest(png, calibration, labels=(), words=(), paper_id=None, figure=None, pdf_page=None)` | Effect, lower, and upper records per forest-plot row, top to bottom. |
| `read_scatter` | `read_scatter(png, calibration, names=('x', 'y'), paper_id=None, figure=None, pdf_page=None)` | ``points`` as calibrated (x, y) pairs and ``records`` of n and Pearson r. |
| `read_flow` | `read_flow(text='', paper_id=None, figure=None, pdf_page=None, words=())` | n records from 'n = N' in a flow diagram, each named by up to five words before it. |

## Agreement

Numeric fields with no valid paired values return empty error and bias metrics. `kappa`
requires equal-length label lists; `icc` requires finite ratings from at least two raters.
Excel exports preserve text beginning with `=` as literal text. Coding-sheet imports
require `paper_id`, `field`, and `coded_value` columns.

| Name | Signature | Purpose |
|---|---|---|
| `AGREEMENT_COLUMNS` | `AGREEMENT_COLUMNS` | Column order of `evaluate`. |
| `evaluate` | `evaluate(extracted, coded, key=('paper_id', 'field'), tolerance=None)` | Per-field agreement between extracted records and a human-coded sheet. |
| `kappa` | `kappa(a, b)` | Cohen's kappa for two raters' labels; None with fewer than two pairs or chance agreement 1. |
| `icc` | `icc(ratings)` | ICC(2,1) from an n targets by k raters array; None with fewer than two targets. |
| `graph_agreement` | `graph_agreement(extracted_rows, manual_rows, y_range=None)` | Point-level error of extracted graph rows against a manual digitizer. |
| `write_coding_sheet` | `write_coding_sheet(records, path)` | Write records as an .xlsx for a second coder: the long columns and an empty ``coded_value``. |
| `read_coding_sheet` | `read_coding_sheet(path)` | Coded records from a filled coding sheet, with the coder's ``coded_value`` as ``value``. |

## Input and export checks

Codebook definitions, queries, and notes must be text; range bounds must be finite.
Model extraction keeps non-finite numbers and invalid passage citations under review.
Retrieval requires a positive integer `top_k` and finite embeddings with matching
dimensions. A retrieval cache tracks document content and encoder identity.

Effect calculations return `not_computed` with named checks for malformed numeric
inputs, fractional sample sizes, negative test statistics, or invalid variance overrides.
Wide exports reject generated column collisions. Matrix exports reject filename
collisions, including the reserved `n.csv`, repeated or reserved `construct` columns,
and asymmetric matrices. Normalized
construct names that match several constructs stay unmapped.
