import pytest

from pubextractor.licensing import check_llm_use
from pubextractor.tables import extract_table_values, read_tables, table_grid

ROWS = [['Outcome', 'Treatment (n = 30)', 'Control (n = 30)'],
        ['Reading', '52.1 (8.0)', '47.3 (8.4)'],
        ['Math', '\u2014', '40.2 (7.7)']]
LOCAL = dict(permission=check_llm_use(access='subscription', ai_terms='silent'),
             endpoint='http://localhost:11434/v1')


def build(tmp_path, name, rules):
    pytest.importorskip('reportlab')
    pytest.importorskip('pdfplumber')
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

    table = Table(ROWS)
    table.setStyle(TableStyle(rules))
    path = tmp_path / name
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [Paragraph('Table 1', getSampleStyleSheet()['Normal']), table])
    return path


def values(rows, **kwargs):
    return extract_table_values([table_grid({'table': 1, 'rows': rows})], paper_id='S01', **kwargs)


class TestApaPdfTable:
    def test_horizontal_rules_only_table_matches_gridded_copy(self, tmp_path):
        apa = build(tmp_path, 'apa.pdf', [('LINEABOVE', (0, 0), (-1, 0), 1, (0, 0, 0)),
                                          ('LINEBELOW', (0, 0), (-1, 0), 1, (0, 0, 0)),
                                          ('LINEBELOW', (0, -1), (-1, -1), 1, (0, 0, 0))])
        gridded = build(tmp_path, 'grid.pdf', [('GRID', (0, 0), (-1, -1), 1, (0, 0, 0))])
        found, expected = read_tables(apa), read_tables(gridded)
        assert len(found) == 1
        assert found[0]['rows'] == expected[0]['rows']
        assert (found[0]['strategy'], expected[0]['strategy']) == ('text', 'lines')

    def test_text_strategy_extracts_only_when_column_counts_agree(self):
        rows = [['Group', 'M'], ['Treatment', '5.0']]
        found = {agree: extract_table_values(
                     [{**table_grid({'table': 1, 'rows': rows}), 'strategy': 'text',
                       'columns_agree': agree}], paper_id='S01')[0]['status']
                 for agree in (True, False)}
        assert found == {True: 'extracted', False: 'review'}

    def test_apa_grid_reports_agreeing_columns(self, tmp_path):
        apa = build(tmp_path, 'apa.pdf', [('LINEABOVE', (0, 0), (-1, 0), 1, (0, 0, 0)),
                                          ('LINEBELOW', (0, 0), (-1, 0), 1, (0, 0, 0)),
                                          ('LINEBELOW', (0, -1), (-1, -1), 1, (0, 0, 0))])
        assert read_tables(apa)[0]['columns_agree'] is True


class TestBlankAndFailedCells:
    def test_blank_body_cell_under_mapped_stat_is_not_found(self):
        records = values([['Outcome', 'Treatment n', 'Control n'], ['Reading', '30', '']])
        blank = next(r for r in records if r['field'] == 'control_n')
        assert (blank['status'], blank['value'], blank['arm'], blank['outcome'], blank['stat']) == (
            'not_found', None, 'Control', 'Reading', 'n')

    def test_rule_mapped_columns_stay_extracted_when_others_are_refused(self):
        records = values([['Measure', 'Treatment M', 'Control M', 'Other'],
                          ['Score', '5.0', '4.0', '3']],
                         respond=lambda prompt, images=None: '{}')
        found = {(r['field'], r['value']): r['status'] for r in records}
        assert found == {('treatment_mean', 5.0): 'extracted', ('control_mean', 4.0): 'extracted',
                         ('unmapped', 3.0): 'refused'}

    def test_failed_model_call_gives_review_with_exception_name(self):
        def respond(prompt, images=None):
            raise RuntimeError('down')
        records = values([['Group', 'Score', 'M'], ['Treatment', '61.3', '50.0']],
                         respond=respond, **LOCAL)
        found = {r['col']: r for r in records}
        assert (found[2]['status'], found[2]['quote']) == ('review', 'model call failed: RuntimeError')
        assert (found[3]['field'], found[3]['status']) == ('treatment_mean', 'extracted')

    def test_candidates_is_one_on_value_records(self):
        records = values([['Group', 'M'], ['Treatment', '61.3']])
        assert {r['candidates'] for r in records} == {1}


class TestCaptionSpacing:
    @pytest.mark.parametrize('caption', ['TABLE1', 'Table1.', 'TABLE 1'])
    def test_caption_without_space_is_read(self, borderless_pdf, caption):
        rows = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
        prose = 'Scores in the tutored group rose by about six points over the whole term.'
        found = read_tables(borderless_pdf([caption, rows, prose]))
        assert [(t['strategy'], t['rows'], t['caption']) for t in found] == [
            ('caption', rows, caption)]
