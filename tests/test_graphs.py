import csv
import io

import numpy as np
import pytest

pytest.importorskip('PIL')
matplotlib = pytest.importorskip('matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from pubextractor import (  # noqa: E402
    SCD_COLUMNS,
    calibrate,
    design_rows,
    figure_images,
    find_phase_lines,
    find_points,
    graph_rows,
    read_graph,
    split_panels,
    to_scdbayes,
)

AB = {'sessions': np.arange(1, 13), 'values': [20, 25, 15, 30, 22, 55, 70, 65, 80, 75, 85, 90],
      'change': 5.5}
MBD = [{'values': [10, 15, 12, 50, 60, 70, 65, 75, 80, 85, 82, 90], 'change': 3.5},
       {'values': [20, 18, 22, 25, 19, 60, 72, 70, 78, 85, 88, 92], 'change': 5.5},
       {'values': [5, 10, 8, 12, 9, 11, 7, 55, 65, 70, 80, 85], 'change': 7.5}]


def _pixel(fig, ax, x, y):
    xd, yd = ax.transData.transform((x, y))
    return xd - 0.5, fig.bbox.height - yd - 0.5


def _draw(ax, values, change, color='k'):
    ax.plot(np.arange(1, 13), values, 'o-', color=color, markersize=7, linewidth=1.5)
    ax.axvline(change, color='k', linestyle='--', linewidth=1.5)
    ax.set(xlim=(0, 13), ylim=(-5, 100), xticks=[0, 5, 10], yticks=[0, 50, 100])
    ax.spines[['top', 'right']].set_visible(False)


def _save(fig):
    fig.canvas.draw()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=fig.dpi)
    plt.close(fig)
    return buffer.getvalue()


def _ticks(fig, ax, offset=0):
    return ([(_pixel(fig, ax, v, 0)[0], v) for v in (0, 10)],
            [(_pixel(fig, ax, 0, v)[1] - offset, v) for v in (0, 100)])


@pytest.fixture
def ab_graph():
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    _draw(ax, AB['values'], AB['change'])
    ax.set(xlabel='Sessions', ylabel='Percent on task')
    ticks = _ticks(fig, ax)
    change = _pixel(fig, ax, AB['change'], 0)[0]
    return _save(fig), ticks, change


@pytest.fixture
def mbd_graph():
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), dpi=100, sharex=True,
                             gridspec_kw={'hspace': 0.35})
    [_draw(ax, case['values'], case['change']) for ax, case in zip(axes, MBD)]
    axes[-1].set_xlabel('Sessions')
    fig.canvas.draw()
    places = [(_ticks(fig, ax), _pixel(fig, ax, case['change'], 0)[0])
              for ax, case in zip(axes, MBD)]
    return _save(fig), places


def _phases(change):
    return ['A1' if s < change else 'B1' for s in range(1, 13)]


class TestCalibrate:
    def test_two_ticks_give_a_linear_map(self):
        axis = calibrate([(100, 0), (300, 10)], [(400, 0), (100, 100)])
        assert axis['x']['slope'] == pytest.approx(0.05)
        assert axis['y']['intercept'] == pytest.approx(400 / 3)
        assert not axis['x']['log'] and axis['x']['residual'] < 1e-9

    def test_even_ticks_over_powers_of_ten_flag_a_log_axis(self):
        axis = calibrate([(0, 0), (10, 1)], [(300, 1), (200, 10), (100, 100), (0, 1000)])
        assert axis['y']['log'] and not axis['x']['log']

    def test_misplaced_tick_shows_in_the_residual(self):
        assert calibrate([(0, 0), (10, 1)], [(0, 0), (50, 5), (100, 20)])['y']['residual'] > 0.1

    def test_one_tick_is_refused(self):
        with pytest.raises(ValueError, match='two or more ticks'):
            calibrate([(5, 1)], [(0, 0), (10, 1)])


class TestAbGraph:
    def test_points_recovered_within_one_percent(self, ab_graph):
        png, (x_ticks, y_ticks), _ = ab_graph
        rows = graph_rows(find_points(png), calibrate(x_ticks, y_ticks))
        assert [r['session'] for r in rows] == list(AB['sessions'])
        assert np.abs(np.array([r['outcome'] for r in rows]) - AB['values']).max() < 1

    def test_dashed_phase_line_is_found(self, ab_graph):
        png, _, change = ab_graph
        assert find_phase_lines(png) == [pytest.approx(change, abs=2)]

    def test_phases_follow_the_line(self, ab_graph):
        png, ticks, _ = ab_graph
        rows = graph_rows(find_points(png), calibrate(*ticks), find_phase_lines(png))
        assert [r['phase'] for r in rows] == _phases(AB['change'])

    def test_color_picks_one_series(self):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        _draw(ax, AB['values'], AB['change'])
        ax.plot(np.arange(1, 13), [5] * 12, 's', color='#d62728', markersize=7)
        png = _save(fig)
        assert len(find_points(png, colour='#d62728')) == 12
        assert len(find_points(png)) == 24


class TestMarkers:
    def _within(self, fig, ax, found, truth):
        expected = np.array([_pixel(fig, ax, x, y) for x, y in truth])
        span = np.abs(np.subtract(_pixel(fig, ax, 13, 100), _pixel(fig, ax, 0, -5)))
        return np.all(np.abs(np.array(found) - expected) <= 0.01 * span)

    @pytest.mark.parametrize('marker', ['o', 's'])
    def test_open_markers_are_found(self, marker):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(AB['sessions'], AB['values'], marker + '-', color='k', mfc='none', markersize=9)
        ax.set(xlim=(0, 13), ylim=(-5, 100), xticks=[0, 10], yticks=[0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        fig.canvas.draw()
        truth = list(zip(AB['sessions'], AB['values']))
        found = find_points(_save(fig))
        assert len(found) == 12 and self._within(fig, ax, found, truth)

    def test_overlapping_markers_split_into_two(self):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        truth = [(2, 20), (4, 60), (6, 30), (7.6, 50), (8, 52), (10, 80)]
        ax.plot(*zip(*truth), 'o', color='k', markersize=12)
        ax.set(xlim=(0, 13), ylim=(-5, 100), xticks=[0, 10], yticks=[0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        fig.canvas.draw()
        found = find_points(_save(fig))
        assert len(found) == 6 and self._within(fig, ax, found, truth)


class TestMultipleBaseline:
    def test_stacked_panels_split_at_blank_rows(self, mbd_graph):
        assert len(split_panels(mbd_graph[0])) == 3

    def test_each_panel_gives_its_case(self, mbd_graph):
        png, places = mbd_graph
        for panel, case, ((x_ticks, y_ticks), change) in zip(split_panels(png), MBD, places):
            calibration = calibrate(x_ticks, [(p - panel['top'], v) for p, v in y_ticks])
            rows = graph_rows(find_points(panel['png']), calibration,
                              find_phase_lines(panel['png']), case=f"case{panel['panel']}",
                              panel=panel['panel'], design='multiple_baseline')
            assert np.abs(np.array([r['outcome'] for r in rows]) - case['values']).max() < 1
            assert [r['phase'] for r in rows] == _phases(case['change'])
            assert find_phase_lines(panel['png']) == [pytest.approx(change, abs=2)]


class TestDesigns:
    def test_ab_reads_one_case_with_two_phases(self, ab_graph):
        png, ticks, _ = ab_graph
        rows = design_rows(read_graph(png, 'ab'), calibrate(*ticks), 'ab', cases=['Ana'])
        assert {r['case'] for r in rows} == {'Ana'}
        assert [r['phase'] for r in rows] == _phases(AB['change'])

    def test_multiple_baseline_shifts_one_calibration_to_each_panel(self, mbd_graph):
        png, places = mbd_graph
        panels = read_graph(png, 'multiple_baseline')
        rows = design_rows(panels, calibrate(*places[0][0]), 'multiple_baseline')
        assert [p['panel'] for p in panels] == [1, 2, 3]
        for number, case in enumerate(MBD, start=1):
            own = [r for r in rows if r['case'] == f"case{number}"]
            assert np.abs(np.array([r['outcome'] for r in own]) - case['values']).max() < 1
            assert [r['phase'] for r in own] == _phases(case['change'])

    def test_panel_count_splits_evenly_when_asked(self, mbd_graph):
        assert len(read_graph(mbd_graph[0], 'multiple_probe', count=3)) == 3

    def test_alternating_series_come_from_marker_shape(self):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        sessions = np.arange(1, 13)
        token, praise = sessions[::2], sessions[1::2]
        ax.plot(token, 60 + token, 'o-', color='k', markersize=8)
        ax.plot(praise, 20 + praise, 's--', color='k', markersize=8)
        ax.set(xlim=(0, 13), ylim=(0, 100), xticks=[0, 10], yticks=[0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ticks = _ticks(fig, ax)
        rows = design_rows(read_graph(_save(fig), 'alternating'), calibrate(*ticks), 'alternating')
        assert [r['session'] for r in rows] == list(sessions)
        assert [r['series'] for r in rows] == ['circle', 'square'] * 6
        assert [r['phase'] for r in rows] == [r['series'] for r in rows]
        expected = [60 + s if s % 2 else 20 + s for s in sessions]
        assert np.abs(np.array([r['outcome'] for r in rows]) - expected).max() < 1

    @pytest.mark.parametrize('style', ['-', '--'])
    def test_changing_criterion_adds_the_criterion_per_phase(self, style):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        values = [10, 14, 8, 12, 31, 28, 34, 30, 52, 49, 55, 51]
        ax.plot(np.arange(1, 13), values, 'o-', color='k', markersize=7)
        [ax.axvline(x, color='k', linestyle='--') for x in (4.5, 8.5)]
        ax.hlines([30, 50], [4.5, 8.5], [8.5, 12.5], color='k', linewidth=1.5, linestyle=style)
        ax.set(xlim=(0, 13), ylim=(0, 100), xticks=[0, 10], yticks=[0, 100])
        ax.spines[['top', 'right']].set_visible(False)
        ticks = _ticks(fig, ax)
        rows = design_rows(read_graph(_save(fig), 'changing_criterion'), calibrate(*ticks),
                           'changing_criterion')
        assert [r['phase'] for r in rows] == ['A1'] * 4 + ['B1'] * 4 + ['A2'] * 4
        assert [r['criterion'] for r in rows[:4]] == [''] * 4
        assert [round(r['criterion']) for r in rows[4:]] == [30] * 4 + [50] * 4


class TestGraphRows:
    CALIBRATION = calibrate([(0, 0), (100, 10)], [(100, 0), (0, 100)])

    @pytest.mark.parametrize('outcome_type, expected', [
        ('count', [3, 48]), ('binary', [1, 1]), ('proportion', [0.032, 0.476]),
        ('ordinal', [3.0, 48.0]), ('continuous', [3.2, 47.6])])
    def test_values_snap_by_outcome_type(self, outcome_type, expected):
        rows = graph_rows([(52, 52.4), (11, 96.8)], self.CALIBRATION, outcome_type=outcome_type)
        assert [r['outcome'] for r in rows] == pytest.approx(expected)
        assert [r['session'] for r in rows] == [1, 5]

    def test_ordinal_snaps_to_listed_levels(self):
        rows = graph_rows([(10, 97)], self.CALIBRATION, outcome_type='ordinal', levels=[0, 5, 10])
        assert rows[0]['outcome'] == 5

    def test_three_lines_give_a_b_a_b(self):
        rows = graph_rows([(5, 50), (15, 50), (25, 50), (35, 50)], self.CALIBRATION,
                          [10, 20, 30])
        assert [r['phase'] for r in rows] == ['A1', 'B1', 'A2', 'B2']

    def test_unknown_design_and_column_are_refused(self):
        with pytest.raises(ValueError, match="design 'abab'.*|unknown columns"):
            graph_rows([(5, 50)], self.CALIBRATION, design='abab', color='red')

    def test_rows_carry_every_column(self):
        row = graph_rows([(15, 50)], self.CALIBRATION, study='S01', case='Ana')[0]
        assert tuple(row) == SCD_COLUMNS and row['confidence'] == 1.0 and row['checks'] == ''

    def test_points_beyond_the_tick_range_are_flagged_and_kept(self):
        rows = graph_rows([(50, 50), (130, 50), (60, -10), (3, 50)], self.CALIBRATION)
        flagged = {r['session']: (r['checks'], r['confidence']) for r in rows}
        assert flagged == {0: ('outside_axes', 0.5), 5: ('', 1.0), 6: ('outside_axes', 0.5),
                           13: ('outside_axes', 0.5)}

    def test_plot_area_widens_the_range_but_not_below_session_one(self):
        rows = graph_rows([(130, 50), (3, 50)], self.CALIBRATION, area=(0, 140, 0, 100))
        assert [(r['session'], r['checks']) for r in rows] == [(0, 'outside_axes'), (13, '')]

    @pytest.mark.parametrize('y_unit, outcome, checks', [
        ('', 6.5, 'check_scale'), ('% correct', 0.065, '')])
    def test_proportion_reads_percent_only_from_a_percent_axis(self, y_unit, outcome, checks):
        calibration = calibrate([(0, 0), (100, 10)], [(100, 0), (0, 10)])
        row = graph_rows([(50, 35)], calibration, outcome_type='proportion', y_unit=y_unit)[0]
        assert (row['outcome'], row['checks']) == (pytest.approx(outcome), checks)


@pytest.mark.parametrize('read', [split_panels, lambda png: read_graph(png, 'multiple_baseline')])
def test_blank_image_has_no_axis_lines(read):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new('RGB', (200, 200), 'white').save(buffer, format='PNG')
    with pytest.raises(ValueError, match='no axis lines found'):
        read(buffer.getvalue())


def test_csv_header_names_case_session_phase(tmp_path):
    rows = graph_rows([(5, 50)], TestGraphRows.CALIBRATION, study='S01', case='Ana')
    with open(to_scdbayes(rows, tmp_path / 'graph.csv'), encoding='utf-8') as handle:
        read = list(csv.DictReader(handle))
    assert tuple(read[0]) == SCD_COLUMNS
    assert read[0]['case'] == 'Ana' and read[0]['phase'] == 'A1'


class TestFigureImages:
    def test_docx_pictures_take_figure_captions(self, ab_graph, tmp_path):
        docx = pytest.importorskip('docx')
        document = docx.Document()
        document.add_paragraph('Results for the first participant.')
        document.add_picture(io.BytesIO(ab_graph[0]))
        document.add_paragraph('Figure 1. Percent of intervals on task.')
        document.save(tmp_path / 'paper.docx')
        found = figure_images(tmp_path / 'paper.docx')
        assert [(f['figure'], f['pdf_page'], f['caption']) for f in found] == [
            (1, None, 'Figure 1. Percent of intervals on task.')]
        assert find_points(found[0]['png'])

    @pytest.mark.parametrize('caption', ['Figure 1.', 'FIGURE1.', 'Fig.1'])
    def test_pdf_figure_is_cropped_above_its_caption(self, ab_graph, tmp_path, caption):
        pytest.importorskip('pdfplumber')
        pytest.importorskip('reportlab')
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        sheet = canvas.Canvas(str(tmp_path / 'paper.pdf'), pagesize=(612, 792))
        sheet.drawString(72, 740, 'The participant was observed in each session of the study.')
        sheet.drawImage(ImageReader(io.BytesIO(ab_graph[0])), 72, 420, width=432, height=288)
        sheet.drawString(72, 400, f'{caption} Percent of intervals on task across sessions.')
        sheet.drawString(72, 370, 'Rates rose after the intervention began in every session.')
        sheet.save()
        found = figure_images(tmp_path / 'paper.pdf')
        assert [(f['figure'], f['pdf_page']) for f in found] == [(1, 1)]
        assert found[0]['caption'].startswith(caption)
        assert len(find_points(found[0]['png'])) == 12


@pytest.mark.parametrize('caption, words, expected', [
    ('Figure 1. Percent of intervals on task', [], ('proportion', ["'Percent' on the graph or caption"])),
    ('Figure 2.', [{'text': 'Number'}, {'text': 'of'}, {'text': 'requests'}],
     ('count', ["'Number' on the graph or caption"])),
    ('Figure 3. Sessions', [], ('continuous', [])),
    ('Figure 4. Levels of prompting across sessions', [], ('continuous', [])),
    ('Figure 5. Words read in the present study', [], ('continuous', []))])
def test_outcome_type_suggestion(caption, words, expected):
    from pubextractor import suggest_outcome_type
    assert suggest_outcome_type(caption, words) == expected
