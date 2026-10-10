import io

import numpy as np
import pytest

pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from pubextractor import (  # noqa: E402
    FIGURE_TYPES,
    build_comparisons,
    calibrate,
    compute_effect,
    figure_images,
    figure_type,
    read_bars,
    read_flow,
    read_forest,
    read_scatter,
)

BARS = {'means': [52, 47], 'se': [3, 4]}
FOREST = {'studies': ['Adams 2019', 'Baker 2020', 'Chen 2021', 'Diaz 2022'],
          'effect': [0.2, 0.55, -0.1, 0.35], 'lower': [-0.15, 0.2, -0.5, 0.05],
          'upper': [0.55, 0.9, 0.3, 0.65]}
SCATTER_X = np.linspace(0.5, 9.5, 15)
SCATTER_Y = 0.8 * SCATTER_X + np.random.default_rng(3).normal(0, 1.5, 15) + 1
FLOW_TEXT = ('Assessed for eligibility (n = 120) Excluded (n = 20) Allocated to intervention '
             '(n = 50) Allocated to control (n = 50) Analysed (n = 48) Analysed (n = 47)')


def _pixel(fig, ax, x, y):
    xd, yd = ax.transData.transform((x, y))
    return xd - 0.5, fig.bbox.height - yd - 0.5


def _save(fig):
    fig.canvas.draw()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=fig.dpi)
    plt.close(fig)
    return buffer.getvalue()


def _bar_chart(errors, **style):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.bar([0, 1], BARS['means'], yerr=errors, capsize=8, width=0.6,
           **(style or {'color': ['#4c72b0', '#aaaaaa']}))
    ax.set(xlim=(-0.6, 1.6), ylim=(0, 60), xticks=[0, 1], yticks=[0, 20, 40, 60])
    ax.spines[['top', 'right']].set_visible(False)
    fig.canvas.draw()
    ticks = ([(_pixel(fig, ax, v, 0)[0], v) for v in (0, 1)],
             [(_pixel(fig, ax, 0, v)[1], v) for v in (0, 60)])
    return _save(fig), calibrate(*ticks)


@pytest.fixture
def bar_chart():
    return _bar_chart(BARS['se'])


@pytest.fixture
def forest_plot():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    rows = np.arange(4)[::-1]
    effect, lower, upper = (np.array(FOREST[key]) for key in ('effect', 'lower', 'upper'))
    ax.errorbar(effect, rows, xerr=[effect - lower, upper - effect], fmt='s', color='k',
                markersize=8, linewidth=1.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=1)
    ax.set(xlim=(-1, 1.2), ylim=(-0.7, 3.7), xticks=[-1, 0, 1], yticks=rows,
           yticklabels=FOREST['studies'])
    fig.canvas.draw()
    ticks = ([(_pixel(fig, ax, v, 0)[0], v) for v in (-1, 1)],
             [(_pixel(fig, ax, 0, v)[1], v) for v in (0, 3)])
    words = [{'x': _pixel(fig, ax, -1, 0)[0] - 60, 'y': _pixel(fig, ax, 0, row)[1], 'text': name}
             for row, name in zip(rows, FOREST['studies'])]
    return _save(fig), calibrate(*ticks), words


@pytest.fixture
def scatter_plot():
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.scatter(SCATTER_X, SCATTER_Y, s=30, color='k')
    ax.set(xlim=(0, 10), ylim=(-2, 14), xticks=[0, 10], yticks=[0, 10])
    ax.spines[['top', 'right']].set_visible(False)
    fig.canvas.draw()
    ticks = ([(_pixel(fig, ax, v, 0)[0], v) for v in (0, 10)],
             [(_pixel(fig, ax, 0, v)[1], v) for v in (0, 10)])
    return _save(fig), calibrate(*ticks)


def _line_graph():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(np.arange(1, 13), [20, 25, 15, 30, 22, 55, 70, 65, 80, 75, 85, 90], 'o-', color='k',
            markersize=7)
    ax.axvline(5.5, color='k', linestyle='--')
    ax.set(xlim=(0, 13), ylim=(-5, 100))
    ax.spines[['top', 'right']].set_visible(False)
    return _save(fig)


def _flow_diagram():
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis('off')
    [ax.annotate(text, (0.5, y), ha='center', bbox={'boxstyle': 'square', 'fc': 'w'})
     for text, y in (('Enrolled (n = 120)', 0.9), ('Allocated (n = 100)', 0.5),
                     ('Analysed (n = 95)', 0.1))]
    [ax.annotate('', (0.5, y - 0.3), (0.5, y - 0.05), arrowprops={'arrowstyle': '->'})
     for y in (0.9, 0.5)]
    return _save(fig)


def _lesson_diagram():
    fig = plt.figure(figsize=(6, 6), dpi=100)
    fig.add_artist(plt.Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, linewidth=2))
    fig.add_artist(plt.Line2D([0.2, 0.2], [0.02, 0.98], color='k', linewidth=2))
    [fig.text(0.6, y, text, ha='center', fontsize=12, bbox={'boxstyle': 'square', 'fc': 'w'})
     for text, y in (('Read the problem', 0.8), ('Draw the model', 0.5), ('Write the answer', 0.2))]
    [fig.add_artist(plt.Arrow(0.6, y, 0, -0.18, width=0.05, color='k')) for y in (0.75, 0.45)]
    return _save(fig)


def _values(records, stat):
    return [r['value'] for r in records if r['stat'] == stat]


class TestFigureType:
    @pytest.mark.parametrize('caption, expected', [
        ('Figure 2. Participant flow through the trial', 'flow'),
        ('Figure 3. Forest plot of study effects', 'forest'),
        ('Figure 1. Posttest means (SE)', 'bar'),
        ('Figure 4. Percent on task across baseline and intervention sessions', 'scd_line')])
    def test_caption_words_decide(self, scatter_plot, caption, expected):
        assert figure_type(_line_graph() if expected == 'scd_line' else scatter_plot[0],
                           caption) == expected

    def test_structure_decides_without_caption_words(self, bar_chart, forest_plot, scatter_plot):
        found = [figure_type(png, 'Figure 1.') for png in (
            bar_chart[0], forest_plot[0], scatter_plot[0], _line_graph(), _flow_diagram())]
        assert found == ['bar', 'forest', 'scatter', 'scd_line', 'flow']

    @pytest.mark.parametrize('design, expected', [('group', 'bar'), ('correlational', 'scatter'),
                                                  ('single_case', 'scd_line'), (None, 'scd_line')])
    def test_confirmed_design_is_a_prior_for_line_graphs(self, design, expected):
        assert figure_type(_line_graph(), 'Figure 1.', design) == expected

    def test_group_paper_gets_a_line_graph_only_from_its_caption(self):
        assert figure_type(_line_graph(), 'Figure 1. Scores across sessions', 'group') == 'scd_line'

    def test_lesson_diagram_in_a_single_case_paper_is_other(self):
        assert figure_type(_lesson_diagram(), 'Figure 1. Instructional sequence in lessons.',
                           'single_case') == 'other'

    def test_line_graph_in_a_single_case_paper_stays_a_line_graph(self):
        assert figure_type(_line_graph(), 'Figure 1. Instructional sequence in lessons.',
                           'single_case') == 'scd_line'

    def test_caption_alone_does_not_make_a_line_graph(self, scatter_plot):
        assert figure_type(_flow_diagram(), 'Figure 2. Probes across sessions') == 'other'
        assert figure_type(scatter_plot[0], 'Figure 2. Probes across sessions') == 'other'

    def test_other_is_a_figure_type(self):
        assert 'other' in FIGURE_TYPES

    def test_single_case_paper_needs_line_evidence_for_a_line_graph(self, bar_chart):
        assert figure_type(bar_chart[0], 'Figure 1.', 'single_case') == 'other'
        assert figure_type(bar_chart[0], 'Figure 1. Posttest means', 'single_case') == 'bar'


class TestBars:
    def test_means_and_se_recovered(self, bar_chart):
        records = read_bars(*bar_chart, names=['Treatment', 'Control'], error='se',
                            paper_id='P1', figure=1)
        assert _values(records, 'mean') == pytest.approx(BARS['means'], abs=0.4)
        assert _values(records, 'se') == pytest.approx(BARS['se'], abs=0.3)
        assert {(r['layer'], r['status'], r['table']) for r in records} == {
            ('cv', 'review', 'figure 1')}
        assert [r['group'] for r in records if r['stat'] == 'mean'] == ['treatment', 'control']

    def test_ci_becomes_se(self):
        records = read_bars(*_bar_chart([1.96 * se for se in BARS['se']]), error='ci')
        assert _values(records, 'se') == pytest.approx(BARS['se'], abs=0.3)
        assert 'se = half-width / 1.96' in records[0]['quote']

    def test_sd_bars_give_sd(self, bar_chart):
        assert _values(read_bars(*bar_chart, error='sd'), 'sd') == pytest.approx(BARS['se'],
                                                                                  abs=0.3)

    def test_names_carry_outcome_and_time(self, bar_chart):
        records = read_bars(*bar_chart, names=['Tutoring / Reading / post', 'Control / Reading'])
        assert {(r['arm'], r['outcome'], r['time']) for r in records} == {
            ('Tutoring', 'Reading', 'post'), ('Control', 'Reading', None)}

    def test_records_feed_comparisons(self, bar_chart):
        records = read_bars(*bar_chart, names=['Treatment', 'Control'], paper_id='P1', figure=1)
        comparison = build_comparisons(records)[0]
        assert comparison['arms']['treatment']['post']['mean'] == pytest.approx(52, abs=0.4)
        assert comparison['arms']['control']['post']['se'] == pytest.approx(4, abs=0.3)

    def test_hatched_bars_are_read(self):
        png, calibration = _bar_chart(BARS['se'], hatch='//', color='w', edgecolor='k')
        records = read_bars(png, calibration)
        assert figure_type(png, 'Figure 1.') == 'bar'
        assert _values(records, 'mean') == pytest.approx(BARS['means'], abs=0.4)
        assert _values(records, 'se') == pytest.approx(BARS['se'], abs=0.3)

    def test_typed_n_gives_posttest_g(self, bar_chart):
        records = read_bars(*bar_chart, names=['Treatment n = 30', 'Control (n = 28)'],
                            paper_id='P1', figure=1)
        assert [(r['arm'], r['value'], r['layer']) for r in records if r['stat'] == 'n'] == [
            ('Treatment', 30, 'human'), ('Control', 28, 'human')]
        found = compute_effect(build_comparisons(records)[0])
        mean, se = (np.array(_values(records, stat)) for stat in ('mean', 'se'))
        sd = se * np.sqrt([30, 28])
        pooled = np.sqrt((29 * sd[0] ** 2 + 27 * sd[1] ** 2) / 56)
        assert found['g'] == pytest.approx((1 - 3 / (4 * 56 - 1)) * (mean[0] - mean[1]) / pooled)
        assert 'sd_from_se' in found['imputed_route']

    def test_bars_without_error_bars_give_no_se(self):
        records = read_bars(*_bar_chart(None), names=['Treatment n = 30', 'Control n = 28'],
                            paper_id='P1', figure=1)
        errors = [r for r in records if r['stat'] == 'se']
        assert [r['value'] for r in errors] == [None, None]
        assert all('no_error_bars' in r['quote'] for r in errors)
        assert _values(records, 'mean') == pytest.approx(BARS['means'], abs=0.4)
        found = compute_effect(build_comparisons(records)[0])
        assert found['g'] is None
        assert 'sd_missing: Treatment (se)' in found['checks']

    def test_unknown_error_type_is_refused(self, bar_chart):
        with pytest.raises(ValueError, match='error'):
            read_bars(*bar_chart, error='iqr')


class TestForest:
    def test_effects_and_limits_recovered(self, forest_plot):
        png, calibration, _ = forest_plot
        records = read_forest(png, calibration, labels=FOREST['studies'])
        stats = ('effect', 'lower', 'upper')
        assert np.abs(np.array([_values(records, s) for s in stats])
                      - np.array([FOREST[s] for s in stats])).max() < 0.03
        assert {(r['layer'], r['status']) for r in records} == {('soma', 'review')}
        assert 'template' not in records[0]
        assert records[0]['field'] == 'adams_2019_effect'

    def test_labels_come_from_the_text_layer(self, forest_plot):
        png, calibration, words = forest_plot
        records = read_forest(png, calibration, words=words)
        assert [r['study'] for r in records if r['stat'] == 'effect'] == FOREST['studies']


class TestScatter:
    def test_pairs_and_r_recovered(self, scatter_plot):
        found = read_scatter(*scatter_plot, paper_id='P1', figure=2)
        x, y = np.array(found['points']).T
        assert len(x) == 15
        assert x == pytest.approx(SCATTER_X, abs=0.1) and y == pytest.approx(SCATTER_Y, abs=0.15)
        r = np.corrcoef(SCATTER_X, SCATTER_Y)[0, 1]
        assert _values(found['records'], 'r') == [pytest.approx(r, abs=0.01)]
        assert _values(found['records'], 'n') == [15]


class TestFlow:
    def test_n_per_box_label(self):
        records = read_flow(FLOW_TEXT, paper_id='P1', figure=2)
        assert _values(records, 'n') == [120, 20, 50, 50, 48, 47]
        assert [r['field'] for r in records[:4]] == [
            'assessed_for_eligibility_n', 'excluded_n', 'allocated_to_intervention_n',
            'allocated_to_control_n']
        assert [r['group'] for r in records[2:4]] == ['treatment', 'control']
        assert {r['status'] for r in records} == {'review'}

    def test_no_text_gives_no_records(self):
        assert read_flow('') == []

    def test_pdf_figure_keeps_its_text_layer(self, tmp_path):
        pytest.importorskip('pdfplumber')
        pytest.importorskip('reportlab')
        from reportlab.pdfgen import canvas
        sheet = canvas.Canvas(str(tmp_path / 'paper.pdf'), pagesize=(612, 792))
        sheet.drawString(72, 740, 'Participants were recruited from four schools in one district.')
        [(sheet.rect(200, y - 8, 160, 26), sheet.drawString(210, y, text))
         for text, y in (('Enrolled (n = 120)', 680), ('Intervention (n = 50)', 620),
                         ('Control (n = 50)', 560), ('Analysed (n = 95)', 500))]
        sheet.drawString(72, 460, 'Figure 2. Participant flow.')
        sheet.drawString(72, 430, 'Attrition was low in both arms of the trial at posttest.')
        sheet.save()
        item = figure_images(tmp_path / 'paper.pdf')[0]
        text = ' '.join(w['text'] for w in item['words'])
        assert _values(read_flow(text), 'n') == [120, 50, 50, 95]

    def test_side_by_side_boxes_keep_their_words(self, tmp_path):
        pytest.importorskip('pdfplumber')
        pytest.importorskip('reportlab')
        from reportlab.pdfgen import canvas
        sheet = canvas.Canvas(str(tmp_path / 'paper.pdf'), pagesize=(612, 792))
        sheet.drawString(72, 740, 'Participants were recruited from four schools in one district.')
        boxes = (('Assessed for', 'eligibility (n = 120)', 220, 660),
                 ('Allocated to', 'intervention (n = 50)', 100, 560),
                 ('Allocated to', 'control (n = 48)', 340, 560))
        [(sheet.rect(x - 10, y - 26, 170, 44), sheet.drawString(x, y, first),
          sheet.drawString(x, y - 14, second)) for first, second, x, y in boxes]
        sheet.drawString(72, 480, 'Figure 2. Participant flow.')
        sheet.drawString(72, 450, 'Attrition was low in both arms of the trial at posttest.')
        sheet.save()
        records = read_flow(words=figure_images(tmp_path / 'paper.pdf')[0]['words'])
        assert [(r['field'], r['value']) for r in records] == [
            ('assessed_for_eligibility_n', 120), ('allocated_to_intervention_n', 50),
            ('allocated_to_control_n', 48)]
