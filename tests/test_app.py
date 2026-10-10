import asyncio
import inspect
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('shiny')
pytest.importorskip('pandas')
pytest.importorskip('yaml')

from pubextractor import app as shiny_app  # noqa: E402

RECORDS = [
    {'paper_id': 'S01', 'field': 'sample_size', 'value': 46, 'layer': 'regex',
     'status': 'review', 'quote': 'The sample was N = 46 students.', 'pdf_page': 1,
     'printed_page': None, 'section': 'Method', 'candidates': 2, 'row': None, 'table': None},
    {'paper_id': 'S01', 'field': 'treatment_mean', 'value': 61.3, 'layer': 'table',
     'status': 'extracted', 'quote': '61.3 (14.2)', 'pdf_page': 1, 'printed_page': None,
     'section': '', 'candidates': 0, 'arm': 'Treatment', 'group': 'treatment',
     'outcome': 'Reading', 'stat': 'mean', 'time': None, 'table': 1, 'row': 3, 'col': 2},
]


class TestModule:
    def test_app_object_exists(self):
        assert isinstance(shiny_app.app, shiny_app.App)
        assert callable(shiny_app.run_app)

    def test_importing_the_package_leaves_shiny_unloaded(self):
        code = 'import sys, pubextractor; print("shiny" in sys.modules)'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=shiny_app._HERE.parent)
        assert result.stdout.strip() == 'False'


class TestSteps:
    @pytest.mark.parametrize('model, expected', [('none', 'Rules only'),
                                                 ('ollama', 'Ollama · gemma3:4b'),
                                                 ('api', 'API · openai')])
    def test_mode_label_names_the_route(self, model, expected):
        assert shiny_app.mode_label(model, 'gemma3:4b', 'openai') == expected

    def test_home_button_opens_the_help_pages(self):
        html = str(shiny_app.home_button(True))
        assert 'Home' in html and '&apos;step&apos;, &apos;home&apos;' in html
        assert 'flow-step home on' in html
        assert '&apos;home_pick&apos;, &apos;codebook&apos;' in str(
            shiny_app.help_view('about', 'home_pick'))

    def test_step_button_shows_a_tick_and_count(self):
        html = str(shiny_app.step_button(0, 'papers', 'Papers', True, True, 2))
        assert '&#10003;' in html and 'flow-step done on' in html and '>2<' in html
        assert 'count' not in str(shiny_app.step_button(2, 'model', 'Model', False, False))


class TestTermEncoder:
    def test_rows_are_unit_length_and_shared_words_score_higher(self):
        encoder = shiny_app.TermEncoder(dims=256)
        chunks = encoder.encode(['students in middle school', 'scores rose after tutoring',
                                 'the weather was cold'])
        query = encoder.encode(['middle school students'])
        assert chunks.shape == (3, 256)
        assert np.allclose(np.linalg.norm(chunks, axis=1), 1)
        scores = chunks @ query.ravel()
        assert scores.argmax() == 0

    def test_empty_text_gives_a_zero_vector(self):
        assert not shiny_app.TermEncoder(dims=16).encode(['']).any()


class TestFrames:
    @pytest.mark.parametrize('column, text, old, expected', [
        ('value', '14.8', 14.2, 14.8),
        ('value', '25', 24, 25),
        ('value', 'n.r.', 1.0, 'n.r.'),
        ('status', 'review', 'extracted', 'review'),
        ('status', 'maybe', 'extracted', 'extracted'),
        ('row', '4', 3, 4),
    ])
    def test_parse_edit_types_the_value(self, column, text, old, expected):
        assert shiny_app.parse_edit(column, text, old) == expected


class TestCodebook:
    def test_default_codebook_loads(self):
        book, problems = shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)
        assert problems == [] and set(book) == {'design', 'sample_size', 'grade_band', 'setting',
                                                   'outcome_domain'}

    def test_problems_are_listed_one_by_one(self):
        book, problems = shiny_app.parse_codebook('a:\n  type: color\nb:\n  type: category\n')
        assert book is None and len(problems) == 2

    def test_broken_yaml_is_a_problem(self):
        assert shiny_app.parse_codebook('a: [unclosed')[1][0].startswith('YAML')

    def test_check_codebook_lists_problems_without_the_prefix(self):
        book, problems = shiny_app.check_codebook({'a': {'type': 'color'}})
        assert book is None and problems[0].startswith('a: type must be')

    def test_terms_text_round_trips(self):
        terms = {'group': ['random assignment', 'control group'], 'single_case': ['multiple baseline']}
        text = 'group: random assignment, control group; single_case: multiple baseline'
        assert shiny_app.format_terms(terms) == text
        assert shiny_app.parse_terms(text) == (terms, [])

    def test_terms_text_skips_values_with_no_terms_and_extra_spaces(self):
        assert shiny_app.format_terms({'a': [], 'b': ['x']}) == 'b: x'
        assert shiny_app.parse_terms('  a :  ; b:  x ,  y ;; ') == ({'b': ['x', 'y']}, [])

    def test_terms_text_without_a_colon_is_a_problem(self):
        terms, problems = shiny_app.parse_terms('group: control group; single_case multiple baseline')
        assert terms == {'group': ['control group']}
        assert len(problems) == 1 and 'single_case multiple baseline' in problems[0]

    def test_rows_round_trip_the_example_codebook(self):
        book = shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)[0]
        rows = shiny_app.codebook_rows(book)
        assert list(rows[0]) == list(shiny_app.ROW_COLUMNS)
        assert rows[1]['range'] == '1-100000' and rows[2]['values'] == 'elementary, middle, high'
        assert shiny_app.rows_to_codebook(rows, book) == (book, [])
        bare, problems = shiny_app.rows_to_codebook(rows)
        assert problems == []
        assert bare == {name: {**field, 'color': None} for name, field in book.items()}

    def test_rows_name_each_problem_and_keep_going(self):
        rows = shiny_app.codebook_rows(shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)[0])
        rows[0].update({'search terms': 'group random', 'type': 'color'})
        rows[1].update({'range': 'many'})
        rows.append({**rows[2], 'field': 'setting'})
        rows.append({**rows[2], 'field': ''})
        book, problems = shiny_app.rows_to_codebook(rows)
        assert book is None
        joined = ' | '.join(problems)
        assert all(text in joined for text in ('design: search terms', 'sample_size: range',
                                               'setting: field name used twice', 'row 7'))

    def test_rows_from_edited_cells_load(self):
        rows = shiny_app.codebook_rows(shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)[0])
        rows[0]['search terms'] += '; group: wait-list'
        rows.append({**dict.fromkeys(shiny_app.ROW_COLUMNS, ''), 'field': 'country', 'type': 'text'})
        book, problems = shiny_app.rows_to_codebook(rows)
        assert problems == [] and 'wait-list' in book['design']['terms']['group']
        assert book['country']['type'] == 'text' and book['country']['query'] is None

    @pytest.mark.parametrize('text, span', [('1-100000', (1, 100000)), ('0.5 to 2', (0.5, 2)),
                                           ('-5--2', (-5, -2)), ('', None)])
    def test_range_text_parses(self, text, span):
        assert shiny_app.parse_range(text) == (span, True)

    def test_range_text_rejects_words(self):
        assert shiny_app.parse_range('many') == (None, False)


BOOK = {
    'design': {'type': 'category', 'values': ['group', 'single_case'], 'color': '#112233',
               'terms': {'group': ['control group', 'randomly assigned'],
                         'single_case': ['multiple baseline']}},
    'grade': {'type': 'category', 'values': ['middle'], 'terms': {'middle': ['middle school']}},
    'place': {'type': 'category', 'values': ['school'], 'terms': {'school': ['school']}},
    'sample_size': {'type': 'integer', 'stat': 'n'},
}


class TestHighlight:
    def book(self):
        return shiny_app.check_codebook(BOOK)[0]

    def test_colors_follow_field_order_in_the_palette(self):
        colors = shiny_app.field_colors(self.book())
        assert list(colors.values()) == list(shiny_app.PALETTE[:4])

    def test_multi_word_terms_match_across_lines_and_hyphens(self):
        text = 'Students were randomly\nassigned; a multiple-baseline design followed.'
        spans = shiny_app.term_spans(text, self.book())
        assert [(text[s['start']:s['end']], s['field'], s['value']) for s in spans] == [
            ('randomly\nassigned', 'design', 'group'),
            ('multiple-baseline', 'design', 'single_case')]

    def test_overlaps_keep_the_longest_match(self):
        spans = shiny_app.term_spans('two middle school classes', self.book())
        assert [(s['field'], s['value']) for s in spans] == [('grade', 'middle')]

    def test_hidden_fields_are_skipped(self):
        spans = shiny_app.term_spans('a middle school', self.book(), shown={'place'})
        assert [(s['field'], s['value']) for s in spans] == [('place', 'school')]

    def test_marked_text_escapes_and_labels(self):
        book = self.book()
        text = '<b> a control group'
        html = shiny_app.marked_text(text, shiny_app.term_spans(text, book),
                                     shiny_app.field_colors(book))
        assert html.startswith('&lt;b&gt; a <mark')
        assert shiny_app.PALETTE[0] in html and 'design: group' in html and html.endswith('</mark>')

    def test_term_counts_tally_every_page(self):
        pages = [{'text': 'control group and control group'}, {'text': 'middle school'},
                 {'rows': [['M', 'SD']]}]
        assert shiny_app.term_counts(pages, self.book()) == [
            {'field': 'design', 'value': 'group', 'count': 2},
            {'field': 'grade', 'value': 'middle', 'count': 1}]


class TestTrace:
    def test_ambiguous_rule_hit_stops_at_regex(self):
        book, _ = shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)
        steps = shiny_app.trace_steps(RECORDS[0], book['sample_size'])
        assert [s['label'] for s in steps] == ['Rules', 'Regex', 'Model', 'Quote', 'Status']
        assert steps[1]['problem'] and not steps[2]['problem']

    def test_table_record_steps(self):
        steps = shiny_app.trace_steps(RECORDS[1])
        assert steps[0]['value'].startswith('Table 1 · row 3 · col 2')
        assert not any(s['problem'] for s in steps)

    def test_human_step_closes_the_chain(self):
        human = [{'paper_id': 'S01', 'field': 'treatment_mean', 'action': 'correct',
                  'coder': 'MS'}, {'paper_id': 'S02', 'field': 'treatment_mean',
                                   'action': 'confirm', 'coder': 'JL'}]
        steps = shiny_app.trace_steps(RECORDS[1], human=human)
        assert (steps[-1]['label'], steps[-1]['value']) == ('Human', 'corrected by MS')
        assert steps[-2]['label'] == 'Status'

    def test_help_lists_trace(self):
        assert 'trace' in shiny_app.HELP_PAGES


class TestFixes:
    def test_marked_text_drops_a_hostile_color(self):
        text = 'a control group'
        spans = shiny_app.term_spans(text, shiny_app.check_codebook(BOOK)[0])
        html = shiny_app.marked_text(text, spans, {'design': 'red" onmouseover="x'})
        assert 'onmouseover' not in html and shiny_app.PALETTE[0] in html

    @pytest.mark.parametrize('value, kept', [('#abc', True), ('#A1B2C3', True), ('red', False),
                                             ('#12345', False), ('#abc;x', False), (None, False)])
    def test_color_check_takes_only_hex(self, value, kept):
        assert (shiny_app.color_or(value, '#000000') == value) is kept

    def test_other_extensions_are_skipped_and_not_papers(self, tmp_path):
        note = tmp_path / 'notes.txt'
        note.write_text('hello')
        paper = shiny_app.read_paper(note)
        assert paper['skipped'] and paper['error'] == 'only PDF, DOCX, or image files are read'
        assert shiny_app.count_papers({'a': paper, 'b': {'skipped': False, 'read': True},
                                       'c': {'skipped': False, 'read': False}}) == 1

    def test_unread_paper_records_the_file_and_reads_nothing(self, tmp_path):
        kept = shiny_app.unread_paper(tmp_path / 'study-s01.pdf', 2048)
        assert kept == {'paper_id': 'study-s01', 'name': 'study-s01.pdf',
                        'path': str(tmp_path / 'study-s01.pdf'), 'size': 2048, 'kind': 'pdf',
                        'pages': [], 'grids': [], 'error': '', 'skipped': False,
                        'has_text': False, 'read': False}
        note = shiny_app.unread_paper(tmp_path / 'notes.txt', 5)
        assert note['skipped'] and note['error'] == 'only PDF, DOCX, or image files are read'

    def test_example_uploads_copy_the_shipped_papers(self, tmp_path):
        uploads = shiny_app.example_uploads(tmp_path)
        assert [u['name'] for u in uploads] == ['study-s01.pdf', 'study-s02.docx']
        assert all(Path(u['datapath']).parent == tmp_path for u in uploads)
        assert [Path(u['datapath']).stat().st_size for u in uploads] == [u['size'] for u in uploads]
        assert shiny_app.upload_summary(uploads)[0] == '2 files ready'

    def test_default_codebook_is_the_shipped_example(self):
        shipped = files('pubextractor').joinpath('examples', 'codebook.yml')
        assert shiny_app.DEFAULT_CODEBOOK == shipped.read_text(encoding='utf-8')
        assert files('pubextractor').joinpath('examples', 'codebook.xlsx').is_file()

    def test_read_paper_marks_the_paper_read(self, tmp_path):
        broken = tmp_path / 'broken.pdf'
        broken.write_bytes(b'not a pdf')
        paper = shiny_app.read_paper(str(broken))
        assert paper['read'] and paper['error']

    @pytest.mark.parametrize('paper, status', [
        ({'skipped': True, 'error': 'x', 'read': False}, 'skipped · only PDF, DOCX, or image'),
        ({'skipped': False, 'error': '', 'read': False}, 'ready'),
        ({'skipped': False, 'error': '', 'read': True}, 'read'),
        ({'skipped': False, 'error': 'bad', 'read': True}, 'error')])
    def test_paper_status_before_and_after_reading(self, paper, status):
        assert shiny_app.paper_status(paper) == status

    def test_values_filter_by_paper(self):
        rows = [{'paper_id': 'A'}, {'paper_id': 'B'}, {'paper_id': 'A'}]
        assert len(shiny_app.rows_for(rows, 'A')) == 2
        assert len(shiny_app.rows_for(rows, shiny_app.ALL_PAPERS)) == 3

    def test_ollama_status_needs_a_local_address(self, monkeypatch):
        monkeypatch.setattr(shiny_app, 'ollama_models', lambda url: pytest.fail('queried'))
        assert shiny_app.ollama_status('http://example.com/v1') == 'enter a local Ollama address'
        monkeypatch.setattr(shiny_app, 'ollama_models', lambda url: ['a', 'b'])
        assert shiny_app.ollama_status('http://localhost:11434/v1') == 'Installed: a, b'
        monkeypatch.setattr(shiny_app, 'ollama_models', lambda url: None)
        assert shiny_app.ollama_status('http://localhost:11434/v1') == 'Ollama not reachable'

    def test_model_choices_carry_size_vision_and_installed(self):
        choices = shiny_app.model_choices(['gemma3:4b', 'tinyllama:latest', 'mine:1b'], 'typed:2b')
        assert choices['gemma3:4b'] == 'gemma3:4b · 3.3 GB · reads images · installed'
        assert choices['gemma3:27b'] == 'gemma3:27b · 17 GB · reads images'
        assert choices['tinyllama'] == 'tinyllama · 0.6 GB · text only · installed'
        assert choices['mine:1b'] == 'mine:1b · installed' and choices['typed:2b'] == 'typed:2b'
        assert list(shiny_app.model_choices()) == list(shiny_app.px.MODEL_SIZES)

    def test_storage_view_warns_only_when_space_is_short(self):
        base = {'message': 'm', 'vision': True, 'installed': False}
        low = str(shiny_app.storage_view({**base, 'enough': False}, 'gemma3:4b'))
        fine = str(shiny_app.storage_view({**base, 'enough': True}, 'gemma3:4b'))
        held = str(shiny_app.storage_view({**base, 'enough': False, 'installed': True},
                                          'gemma3:4b'))
        assert 'text-danger' in low and 'text-danger' not in fine and 'text-danger' not in held
        assert 'To install: ollama pull gemma3:4b' in low and 'ollama pull' not in held
        assert 'reads images' in fine

    def test_guarded_turns_an_exception_into_a_notice(self, monkeypatch):
        shown = []
        monkeypatch.setattr(shiny_app.ui, 'notification_show', lambda text, **kw: shown.append(text))

        @shiny_app.guarded('Reading')
        def broken():
            raise RuntimeError('bad zip')

        broken()
        assert shown == ['Reading failed: RuntimeError: bad zip']

    def test_guarded_keeps_an_async_handler_async(self, monkeypatch):
        shown = []
        monkeypatch.setattr(shiny_app.ui, 'notification_show', lambda text, **kw: shown.append(text))

        @shiny_app.guarded('Reading')
        async def broken():
            raise RuntimeError('bad zip')

        assert inspect.iscoroutinefunction(broken)
        asyncio.run(broken())
        assert shown == ['Reading failed: RuntimeError: bad zip']

    def test_read_paper_opens_the_pdf_once(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(shiny_app.px, 'read_document',
                            lambda path: calls.append(path) or ([{'page': 1, 'text': 'x'}], []))
        monkeypatch.setattr(shiny_app.px, 'read_pdf', lambda path: pytest.fail('read twice'))
        paper = shiny_app.read_paper(tmp_path / 'study.pdf')
        assert len(calls) == 1 and paper['has_text'] and paper['pages'][0]['printed_page'] is None


class TestMessages:
    @pytest.mark.parametrize('size, text', [(512, '512 B'), (48 * 1024, '48 KB'),
                                            (int(1.5 * 1024 ** 2), '1.5 MB'), (None, '0 B')])
    def test_human_size(self, size, text):
        assert shiny_app.human_size(size) == text

    def test_upload_summary_is_short_and_lists_each_file(self):
        uploads = [{'name': 'study-s01.pdf', 'size': 48 * 1024},
                   {'name': 'study-s02.docx', 'size': 37 * 1024},
                   {'name': 'notes.txt', 'size': 5}]
        message, items = shiny_app.upload_summary(uploads)
        assert message == '2 files ready · 1 skipped'
        assert [(i['name'], i['size'], i['status']) for i in items] == [
            ('study-s01.pdf', '48 KB', 'ready'), ('study-s02.docx', '37 KB', 'ready'),
            ('notes.txt', '5 B', 'skipped')]
        assert shiny_app.upload_summary(uploads[:2])[0] == '2 files ready'

    def test_upload_summary_without_readable_files(self):
        message, items = shiny_app.upload_summary([{'name': 'a.txt', 'size': 2048}])
        assert message == 'No files ready · 1 skipped'
        assert items[0]['status'] == 'skipped'

    @pytest.mark.parametrize('name, problem', [
        ('notes.txt', 'unsupported file type .txt: use .xlsx, .csv, or .yml'),
        ('book', 'unsupported file type (none): use .xlsx, .csv, or .yml'),
        ('codebook.YAML', None), ('codebook.xlsx', None)])
    def test_codebook_type_problem(self, name, problem):
        assert shiny_app.codebook_type_problem(name) == problem

    def test_extract_summary_counts_values_and_review(self):
        records = [{'status': 'extracted'}] * 18 + [{'status': 'review'}] * 3 + [
            {'status': 'not_found'}]
        assert (shiny_app.extract_summary(records, 2)
                == 'Extracted 21 values from 2 papers (3 need review)')
        assert (shiny_app.extract_summary([{'status': 'review'}], 1)
                == 'Extracted 1 value from 1 paper (1 needs review)')

    def test_effects_summary_counts_missing_g(self):
        effects = [{'g': 0.4}, {'g': 0.1}, {'g': -0.2}, {'g': None}, {'g': None}]
        assert shiny_app.effects_summary(effects) == 'Computed 3 effect sizes (2 not computed)'

    EFFECTS = [{'status': 'computed', 'checks': []},
               {'status': 'review', 'checks': ['n_from_other_source']},
               {'status': 'not_computed', 'checks': ['values_review', 'n_missing']},
               {'status': 'not_computed', 'checks': ['n_missing']},
               {'status': 'not_computed', 'checks': ['sd_missing: A (sd)', 'no_method']},
               {'status': 'not_computed', 'checks': ['test_statistic_unpaired']}]

    def test_not_computed_rows_collapse_into_a_note(self):
        rows, note = shiny_app.split_effects(self.EFFECTS)
        assert [e['status'] for e in rows] == ['computed', 'review']
        assert note == '4 not computed: 2 n_missing, 1 no_method, 1 test_statistic_unpaired'

    def test_show_all_keeps_every_row(self):
        rows, note = shiny_app.split_effects(self.EFFECTS, show_all=True)
        assert rows == self.EFFECTS and note.startswith('4 not computed')

    def test_no_note_when_every_effect_computes(self):
        assert shiny_app.split_effects(self.EFFECTS[:2])[1] == ''

    def test_text_sizes_name_arms_of_the_paper(self):
        found = [{'paper_id': 'S01', 'arm': 'Tutoring', 'group': 'treatment'},
                 {'paper_id': 'S01', 'arm': 'Usual', 'group': 'control'},
                 {'paper_id': 'S02', 'arm': 'Peer', 'group': 'treatment'}]
        pages = [{'page': 2, 'text': 'In all, N = 46 pupils: tutoring (n = 24), usual (n = 22).',
                  'printed_page': None, 'section': 'Method'}]
        records, total = shiny_app.text_sizes(found, pages, 'S01')
        assert [(r['arm'], r['value'], r['pdf_page'], r['section']) for r in records] == [
            ('Tutoring', 24, 2, 'Method'), ('Usual', 22, 2, 'Method')]
        assert total == 46

    def test_extract_paper_reads_one_paper(self, study_pdf):
        path = study_pdf([['Group', 'n', 'M', 'SD'], ['Treatment', '20', '58.0', '10.0'],
                          ['Control', '21', '52.0', '9.0']], caption=['Table 1. Posttest scores'])
        book = shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)[0]
        paper, found, design, failed = shiny_app.extract_paper(
            shiny_app.unread_paper(path, 10), book, 'study', {}, None,
            shiny_app.px.ProvenanceTracker('fulltext', 'extraction'))
        assert paper['read'] and paper['size'] == 10 and failed == []
        assert {r['paper_id'] for r in found} == {'study'} and len(design) == 2

    def test_design_failure_is_reported_not_raised(self, study_pdf, monkeypatch):
        path = study_pdf([['Group', 'n'], ['Treatment', '20']], caption=['Table 1. Scores'])
        monkeypatch.setattr(shiny_app.px, 'suggest_design', lambda *_: 1 / 0)
        book = shiny_app.parse_codebook(shiny_app.DEFAULT_CODEBOOK)[0]
        design, failed = shiny_app.extract_paper(
            shiny_app.unread_paper(path, 10), book, 'study', {}, None,
            shiny_app.px.ProvenanceTracker('fulltext', 'extraction'))[2:]
        assert design == ('group', ['design not read'])
        assert failed == [f"{path.name}: ZeroDivisionError: division by zero"]

    def test_image_file_is_read_not_skipped(self, tmp_path):
        from PIL import Image
        path = tmp_path / 'table.png'
        Image.new('RGB', (200, 100), 'white').save(path)
        paper = shiny_app.read_paper(path)
        assert not paper['skipped'] and paper['error'] == ''
        assert [g['strategy'] for g in paper['grids']] == ['image']

    def test_saved_and_loaded_messages(self):
        assert shiny_app.saved_message('pubextractor-long.csv', 1) == (
            'Saved pubextractor-long.csv (1 row)')
        assert shiny_app.codebook_loaded('codebook.xlsx', 12 * 1024, 5) == (
            'Codebook loaded: codebook.xlsx (12 KB), 5 fields')

    def test_model_message_names_the_route_or_the_missing_server(self):
        url = 'http://localhost:11434/v1'
        assert shiny_app.model_message('ollama', 'gemma3:4b', 'openai', url, ['gemma3:4b']) == (
            'Model: Ollama · gemma3:4b', 'message')
        assert shiny_app.model_message('ollama', 'gemma3:4b', 'openai', url, None) == (
            f'Ollama not reachable at {url}', 'warning')
        assert shiny_app.model_message('none', '', '', url, None) == ('Model: Rules only',
                                                                     'message')

    def test_attempt_returns_the_error_instead_of_raising(self):
        assert shiny_app.attempt(lambda a, b=0: [a + b], 1, b=2) == ([3], '')
        assert shiny_app.attempt(int, 'x')[0] == []
        assert shiny_app.attempt(int, 'x')[1].startswith('ValueError: ')


MANUAL = [{'paper_id': 'S01', 'field': 'treatment_post_mean', 'value': 20, 'layer': 'human',
           'status': 'extracted', 'quote': 'entered by coder', 'table': 3, 'row': None,
           'col': None, 'stat': 'mean', 'arm': 'CP', 'group': 'treatment', 'outcome': 'Test',
           'time': 'post', 'pdf_page': 9}]
FLAG = {'paper_id': 'S01', 'field': 'table', 'value': None, 'layer': 'table',
        'status': 'not_found', 'quote': 'table 3 on page 9 is an image', 'table': 3, 'row': None}


class TestImageTables:
    @pytest.mark.parametrize('count, text', [(0, ''), (1, '1 image table'), (2, '2 image tables')])
    def test_badge_counts_image_tables(self, count, text):
        grids = [{'strategy': 'image'}] * count + [{'strategy': 'lines'}]
        assert shiny_app.image_badge(grids) == text

    def test_badge_column_takes_tags_in_a_string_frame(self):
        frame = shiny_app.badge_column(shiny_app._frame(
            [{'image tables': '1 image table'}, {'image tables': ''}], ('image tables',)))
        assert 'badge' in str(frame['image tables'][0]) and frame['image tables'][1] == ''

    def test_blank_entries_have_the_entry_columns(self):
        rows = shiny_app.blank_entries()
        assert len(rows) == 4
        assert all(row == dict.fromkeys(shiny_app.ENTRY_COLUMNS, '') for row in rows)
        assert shiny_app.ENTRY_COLUMNS[:6] == ('arm', 'group', 'outcome', 'time', 'stat', 'value')

    def test_model_rows_prefill_marked_review(self):
        records = [{**MANUAL[0], 'layer': 'vision', 'status': 'review', 'quote': '20 (4)'},
                   {**FLAG, 'layer': 'vision', 'status': 'review', 'stat': None}]
        rows = shiny_app.entry_rows(records)
        assert rows == [{'arm': 'CP', 'group': 'treatment', 'outcome': 'Test', 'time': 'post',
                         'stat': 'mean', 'value': '20', 'check': 'review'}]

    def test_saved_values_replace_earlier_entries(self):
        older = [{**MANUAL[0], 'value': 19}, FLAG, {**MANUAL[0], 'table': 4}]
        kept = shiny_app.replace_entries(older, 'S01', 3, MANUAL)
        assert [r['value'] for r in kept if r['table'] == 3 and r.get('stat')] == [20]
        assert FLAG in kept and {**MANUAL[0], 'table': 4} in kept

    def test_saved_vision_values_replace_earlier_entries(self):
        older = [{**MANUAL[0], 'value': 19, 'layer': 'vision'}, FLAG]
        kept = shiny_app.replace_entries(older, 'S01', 3, MANUAL)
        assert [r['value'] for r in kept if r['table'] == 3 and r.get('stat')] == [20]

    def test_new_run_tracker_keeps_human_steps(self):
        old = shiny_app.px.ProvenanceTracker('fulltext', 'extraction')
        shiny_app.px.log_human(old, 'S01', 'table 3', 'enter', None, 6, 'MS')
        old.log_prompt('earlier prompt', model='m')
        new = shiny_app.run_tracker(old)
        assert new is not old
        assert new.to_dict()['human'] == old.to_dict()['human']
        assert 'prompts' not in new.to_dict()

    def test_entered_values_count_as_table_records(self):
        assert shiny_app.is_table_record(MANUAL[0])
        assert shiny_app.is_table_record({**MANUAL[0], 'layer': 'soma', 'table': 'figure 3'})
        assert not shiny_app.is_table_record(RECORDS[0])

    def test_crop_is_inline_png(self):
        html = str(shiny_app.crop_tag(bytes([137]) + b'PNG'))
        assert 'src="data:image/png;base64,iVBORw==' in html

    @pytest.mark.parametrize('model, ollama, api, ready', [
        ('none', 'gemma3:4b', 'gpt', False), ('ollama', 'gemma3:4b', '', True),
        ('ollama', '', 'gpt', False), ('api', '', 'gpt', True)])
    def test_vision_needs_a_model(self, model, ollama, api, ready):
        assert shiny_app.vision_ready(model, ollama, api) is ready


class TestKeepHuman:
    NEW = [RECORDS[1], {**RECORDS[1], 'row': 4, 'value': 7.0}]

    def test_edited_cell_survives_rerun(self):
        edited = {**RECORDS[1], 'value': 99, 'layer': 'human'}
        kept = shiny_app.keep_human(self.NEW, [edited, RECORDS[0]])
        assert [r['value'] for r in kept] == [99, 7.0]

    def test_hand_entered_image_values_are_added_back(self):
        kept = shiny_app.keep_human(self.NEW, MANUAL + [FLAG])
        assert kept == self.NEW + MANUAL

    def test_vision_values_and_accepted_status_carry_over(self):
        seen = {**RECORDS[1], 'layer': 'vision', 'status': 'review'}
        kept = shiny_app.keep_human(self.NEW, [seen])
        assert seen in kept and RECORDS[1] not in kept

    def test_new_records_without_human_counterpart_stay(self):
        assert shiny_app.keep_human(self.NEW, []) == self.NEW


class TestArmRoles:
    ARMS = [{**RECORDS[1], 'arm': 'Tutored', 'group': None, 'stat': 'n', 'value': 20},
            {**RECORDS[1], 'arm': 'Tutored', 'group': None},
            {**RECORDS[1], 'arm': 'Usual', 'group': 'other'},
            {**RECORDS[1], 'table': 2}]

    def test_open_arms_lists_each_unsettled_arm_once(self):
        assert shiny_app.open_arms(self.ARMS + RECORDS) == [('S01', 1, 'Tutored', None),
                                                            ('S01', 1, 'Usual', 'other')]

    def test_apply_roles_sets_group_and_human_source(self):
        roles = {('S01', 1, 'Tutored'): 'treatment', ('S01', 1, 'Usual'): 'control'}
        kept = shiny_app.apply_roles(self.ARMS, roles)
        assert [(r['group'], r.get('group_source')) for r in kept] == [
            ('treatment', 'human'), ('treatment', 'human'), ('control', 'human'),
            ('treatment', None)]
        assert {r['layer'] for r in kept} == {'table'}


class TestAcceptTable:
    ROWS = [{**RECORDS[1], 'table': 1, 'status': 'review', 'layer': 'table'},
            {**RECORDS[1], 'table': 1, 'status': 'not_found', 'layer': 'table'},
            {**RECORDS[1], 'table': 2, 'status': 'review', 'layer': 'table'}]

    def test_review_values_of_one_table_become_human_extracted(self):
        kept, count = shiny_app.accept_table(self.ROWS, 'S01', 1)
        assert count == 1
        assert [(r['status'], r['layer']) for r in kept] == [
            ('extracted', 'human'), ('not_found', 'table'), ('review', 'table')]

    def test_no_table_accepts_every_table_of_the_paper(self):
        rows = self.ROWS + [{**self.ROWS[0], 'paper_id': 'S02'}, {**RECORDS[0], 'status': 'review'}]
        kept, count = shiny_app.accept_table(rows, 'S01')
        assert count == 2
        assert [r['status'] for r in kept] == ['extracted', 'not_found', 'extracted', 'review',
                                               'review']


class TestReviewQueue:
    def test_review_count_per_paper(self):
        rows = [{**RECORDS[0], 'paper_id': 'S02'}] + RECORDS
        assert shiny_app.review_count(rows, 'S01') == (1, 2)
        assert shiny_app.review_count(rows, 'S03') == (0, 0)

    @pytest.mark.parametrize('suggested, confirmed, text', [
        ('group', None, 'group'), ('group', 'single_case', 'single_case ✓'), (None, None, '')])
    def test_design_label_ticks_a_confirmed_design(self, suggested, confirmed, text):
        assert shiny_app.design_label(suggested, confirmed) == text

    def test_queue_row_prefills_the_design_and_counts_review(self):
        row = str(shiny_app.queue_row('S01', 'single_case', False, 3, True))
        assert '<option value="single_case" selected="">' in row
        assert 'Confirm design' in row and '3 to review' in row and 'Accept all tables' in row
        assert 'queue_pick' in row and 'queue_design' in row and 'queue-row on' in row

    def test_confirmed_row_shows_a_tick(self):
        row = str(shiny_app.queue_row('S01', 'group', True, 0, False))
        assert '&#10003;' in row and 'queue-row on' not in row


class TestHelp:
    @pytest.mark.parametrize('name', list(shiny_app.HELP_PAGES))
    def test_every_page_renders_with_a_heading(self, name):
        assert shiny_app.help_html(name).startswith('<h1>')

    def test_tables_and_outside_links_render(self):
        page = shiny_app.help_html('local_model')
        assert '<table>' in page and 'gemma3:4b' in page
        assert '<a target="_blank" rel="noopener" href="https://ollama.com' in page

    def test_unknown_page_falls_back_to_about(self):
        assert shiny_app.help_html('../../.env') == shiny_app.help_html('about')

    def test_view_marks_the_chosen_page(self):
        view = str(shiny_app.help_view('codebook'))
        assert 'class="help-item on"' in view and 'Search terms' in view

    def test_info_button_names_its_page(self):
        assert '&apos;license_check&apos;' in str(shiny_app.info_button('license_check'))


class TestAgreement:
    def test_table_shows_kappa_or_icc_rounded(self):
        rows = [{'field': 'design', 'n_pairs': 6, 'agreement': 200 / 3, 'kappa': 1 / 3, 'icc': None,
                 'mae': None, 'bias': None, 'lower': None, 'upper': None,
                 'hallucinations': 1, 'omissions': 0},
                {'field': 'n', 'n_pairs': 4, 'agreement': 25.0, 'kappa': None, 'icc': 44 / 53,
                 'mae': 0.75, 'bias': -0.75, 'lower': -1.73, 'upper': 0.23,
                 'hallucinations': 0, 'omissions': 2}]
        frame = shiny_app.agreement_table(rows)
        assert list(frame.columns) == ['field', 'n', 'agreement %', 'kappa / ICC', 'MAE',
                                       'hallucinations', 'omissions']
        assert frame.values.tolist() == [['design', 6, 66.7, 'κ 0.33', '', 1, 0],
                                         ['n', 4, 25.0, 'ICC 0.83', 0.75, 0, 2]]

    def test_help_page_is_listed(self):
        assert 'Agreement' in shiny_app.help_html('evaluation')


class TestGraphHelpers:
    def test_click_near_a_point_removes_it_and_elsewhere_adds_one(self):
        points = [(10.0, 10.0), (50.0, 50.0)]
        assert shiny_app.toggle_point(points, 12, 9) == [(50.0, 50.0)]
        assert shiny_app.toggle_point(points, 30, 30) == [(10.0, 10.0), (30, 30), (50.0, 50.0)]

    def test_calibrate_clicks_fill_x_then_y_and_restart(self):
        ticks = {}
        for x, y in [(1, 2), (3, 4), (5, 6), (7, 8)]:
            ticks = shiny_app.set_tick(ticks, x, y)
        assert ticks == {'x1': 1, 'x2': 3, 'y1': 6, 'y2': 8}
        assert shiny_app.set_tick(ticks, 9, 10) == {'x1': 9}

    def test_figures_help_page_is_listed(self):
        assert 'Single-Case Graphs' in shiny_app.help_html('figures')

    def test_added_point_takes_the_nearest_series(self):
        points = [(10.0, 10.0, 'circle'), (50.0, 50.0, 'square')]
        assert (45, 40, 'square') in shiny_app.toggle_point(points, 45, 40)

    def test_replot_draws_a_panel_per_case(self):
        rows = [{'case': case, 'session': s, 'outcome': s, 'series': 1,
                 'phase': 'A1' if s < 3 else 'B1'} for case in ('a', 'b') for s in range(1, 5)]
        assert 'xaxis2' in str(shiny_app.replot_figure(rows))

    def test_replot_breaks_the_line_at_each_phase_change(self, monkeypatch):
        monkeypatch.setattr(shiny_app, '_graph_html', lambda figure, height: figure)
        rows = [{'case': 'a', 'session': s, 'outcome': s, 'series': 1, 'phase': phase}
                for s, phase in zip(range(1, 7), ['A1', 'A1', 'B1', 'B1', 'A2', 'A2'])]
        lines = [trace for trace in shiny_app.replot_figure(rows).data if trace.name == '1']
        assert [list(trace.x) for trace in lines] == [[1, 2], [3, 4], [5, 6]]
        assert [trace.showlegend for trace in lines] == [True, False, False]

    def test_page_head_names_an_inline_favicon(self):
        from htmltools import HTMLDocument
        head = HTMLDocument(shiny_app.app_ui).render()['html'].split('</head>')[0]
        assert '<link rel="icon" href="data:image/svg+xml' in head

    PANELS = [{'panel': 1, 'top': 0, 'bottom': 100, 'axis': 90.0, 'lines': [], 'criteria': [],
               'points': [(10.0, 50.0, 'circle'), (20.0, 50.0, 'square')]},
              {'panel': 2, 'top': 101, 'bottom': 200, 'axis': 190.0, 'lines': [], 'criteria': [],
               'points': []}]

    def test_edited_points_survive_a_new_read(self):
        edited = [(12.0, 40.0), (21.0, 60.0, 'circle'), (30.0, 150.0)]
        placed = shiny_app.place_points(self.PANELS, edited, 'alternating')
        assert placed[0]['points'] == [(12.0, 40.0, 'circle'), (21.0, 60.0, 'square')]
        assert placed[1]['points'] == [(30.0, 150.0, 'square')]
        plain = shiny_app.place_points(self.PANELS, edited, 'ab')
        assert [p['points'] for p in plain] == [[(12.0, 40.0), (21.0, 60.0)], [(30.0, 150.0)]]

    def test_renamed_series_become_series_and_phase(self):
        assert shiny_app.series_names(self.PANELS) == ['circle', 'square']
        panels = shiny_app.name_series(self.PANELS, {'circle': 'Video modeling'})
        rows = shiny_app.px.design_rows(panels, shiny_app.px.calibrate(
            [(0, 0), (10, 1)], [(100, 0), (0, 100)]), 'alternating')
        assert [(r['series'], r['phase']) for r in rows] == [
            ('Video modeling', 'Video modeling'), ('square', 'square')]


def test_keep_human_keeps_row_order():
    from pubextractor.app import keep_human
    new = [{'paper_id': 'p', 'field': f, 'table': 1, 'row': i, 'col': 1, 'value': 1, 'layer': 'table'}
           for i, f in enumerate('abc')]
    old = [{**new[1], 'value': 9, 'layer': 'human'}]
    assert [r['value'] for r in keep_human(new, old)] == [1, 9, 1]
