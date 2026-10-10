import json

import pytest

from pubextractor.provenance import ProvenanceTracker, log_human
from pubextractor.trace import (
    apply_decisions,
    prisma_items,
    read_decisions,
    trace_graph,
    write_trace,
)


def record(field, value, layer='regex', status='extracted', **extra):
    return {'paper_id': 'S01', 'field': field, 'value': value, 'layer': layer,
            'status': status, 'quote': f'{field} is {value}', 'pdf_page': 2,
            'printed_page': '14', 'section': 'Method', 'candidates': 1, **extra}


def arm(field, stat, value, name, group):
    return record(field, value, 'table', table=1, row=2, col=3, arm=name, group=group,
                  outcome='Reading', stat=stat, time='post')


RECORDS = [
    record('grade', 'middle'),
    record('setting', None, 'prompt', 'refused', quote='license forbids'),
    record('duration', 12, 'prompt', 'review'),
    record('country', None, 'none', 'not_found'),
    arm('n_t', 'n', 20, 'A', 'treatment'),
    arm('m_t', 'mean', 61.3, 'A', 'treatment'),
    arm('n_c', 'n', 21, 'B', 'control'),
    {'paper_id': 'S02', 'var_i': 'x', 'var_j': 'y', 'r': 0.4, 'n': 80, 'table': 2,
     'pdf_page': 5, 'status': 'extracted', 'checks': ''},
]
EFFECT = {'paper_id': 'S01', 'es_id': 'S01-1', 'table': 1, 'outcome': 'Reading',
          'treatment': 'A', 'control': 'B', 'g': 0.4, 'var_g': 0.1,
          'imputed_route': 'sd_from_se', 'method': 'smd_post', 'status': 'computed'}
POINTS = [{'study': 'S01', 'figure': '1', 'case': 'Ann', 'session': s, 'outcome': 3}
          for s in (1, 2)]


@pytest.fixture
def tracker():
    tracker = ProvenanceTracker('fulltext', 'extraction')
    tracker.log_permission({'verdict': 'local_only'}, paper_id='S01')
    tracker.log_prompt('q', model='gemma3:4b')
    return tracker


class TestHumanSteps:
    def test_step_has_coder_time_and_both_values(self, tracker):
        step = log_human(tracker, 'S01', 'duration', 'correct', 12, 14, coder='MS')
        assert tracker.to_dict()['human'] == [step]
        assert (step['before'], step['after'], step['coder']) == (12, 14, 'MS')
        assert step['layer'] == 'human' and step['time']

    def test_no_tracker_still_returns_the_step(self):
        assert log_human(None, 'S01', 'grade', 'confirm', 'middle', 'middle')['coder'] is None


class TestTraceGraph:
    def test_field_status(self, tracker):
        status = trace_graph(RECORDS, tracker=tracker)['S01']['status']
        assert status == {'grade': 'confirmed', 'setting': 'refused', 'duration': 'review',
                          'country': 'missing', 'n_t': 'confirmed', 'm_t': 'confirmed',
                          'n_c': 'confirmed'}

    def test_value_chain_from_source_to_export(self):
        graph = trace_graph(RECORDS[:1])['S01']
        steps = [n['step'] for n in graph['nodes'] if n['level'] == 'value']
        assert steps == ['source', 'layer', 'candidates', 'choice', 'checks', 'imputation',
                         'export']
        assert {'from': 'study', 'to': 'field:grade'} in graph['edges']
        assert {'from': 'field:grade', 'to': 'v0.source'} in graph['edges']
        assert {'from': 'v0.imputation', 'to': 'v0.export'} in graph['edges']

    def test_evidence_carries_quote_page_and_license(self, tracker):
        evidence = trace_graph(RECORDS, tracker=tracker)['S01']['evidence']['v0']
        assert (evidence['quote'], evidence['pdf_page'], evidence['license']) == (
            'grade is middle', 2, 'local_only')

    def test_inputs_feed_g_then_variance(self):
        graph = trace_graph(RECORDS, effects=[EFFECT])['S01']
        into_g = {e['from'] for e in graph['edges'] if e['to'] == 'g0'}
        assert into_g == {'v4.export', 'v5.export', 'v6.export'}
        assert {'from': 'g0', 'to': 'vg0'} in graph['edges']
        assert next(n for n in graph['nodes'] if n['id'] == 'g0')['route'] == 'sd_from_se'

    def test_points_feed_scdbayes_rows(self):
        graph = trace_graph([], graph_rows=POINTS)['S01']
        assert {'from': 'points:1', 'to': 'scdbayes'} in graph['edges']
        assert next(n for n in graph['nodes'] if n['id'] == 'scdbayes')['value'] == 2

    def test_matrix_feeds_matrix_files(self):
        graph = trace_graph(RECORDS)['S02']
        assert {'from': 'matrix:2', 'to': 'matrices'} in graph['edges']
        assert graph['status'] == {}

    def test_human_steps_close_the_chain(self, tracker):
        log_human(tracker, 'S01', 'duration', 'correct', 12, 14, coder='MS')
        graph = trace_graph(RECORDS, tracker=tracker)['S01']
        assert {'from': 'v2.export', 'to': 'v2.human0'} in graph['edges']

    def test_write_trace_is_json(self, tmp_path):
        path = write_trace(trace_graph(RECORDS), tmp_path / 'trace.json')
        assert set(json.loads(path.read_text(encoding='utf-8'))) == {'S01', 'S02'}


class TestDecisions:
    DECISIONS = [{'paper_id': 'S01', 'field': 'duration', 'action': 'correct', 'value': 14,
                  'coder': 'MS', 'time': '2026-10-08T10:00:00+00:00'},
                 {'paper_id': 'S01', 'field': 'grade', 'action': 'confirm', 'coder': 'MS'},
                 {'paper_id': 'S01', 'field': 'setting', 'action': 'not_reported'}]

    def test_read_decisions(self, tmp_path):
        path = tmp_path / 'decisions.json'
        path.write_text(json.dumps(self.DECISIONS), encoding='utf-8')
        assert read_decisions(path) == self.DECISIONS

    def test_unknown_action_is_refused(self, tmp_path):
        path = tmp_path / 'decisions.json'
        path.write_text(json.dumps([{'paper_id': 'S01', 'field': 'x', 'action': 'guess'}]),
                        encoding='utf-8')
        with pytest.raises(ValueError, match='guess'):
            read_decisions(path)

    @pytest.mark.parametrize('content, problem', [
        ({'paper_id': 'S01', 'field': 'x', 'action': 'confirm'}, 'list'),
        ([{'field': 'x', 'action': 'confirm'}], 'paper_id'),
        ([{'paper_id': 'S01', 'action': 'confirm'}], 'field'),
        (['confirm'], 'object')])
    def test_malformed_decisions_are_refused(self, tmp_path, content, problem):
        path = tmp_path / 'decisions.json'
        path.write_text(json.dumps(content), encoding='utf-8')
        with pytest.raises(ValueError, match=problem):
            read_decisions(path)

    def test_apply_sets_value_layer_and_status(self, tracker):
        updated, steps = apply_decisions(RECORDS, self.DECISIONS, tracker)
        changed = {r['field']: r for r in updated[:3]}
        assert (changed['duration']['value'], changed['duration']['status']) == (14, 'extracted')
        assert changed['setting']['status'] == 'not_found'
        assert {r['layer'] for r in changed.values()} == {'human'}
        assert steps[0]['time'] == '2026-10-08T10:00:00+00:00'
        assert (steps[0]['before'], steps[0]['after']) == (12, 14)
        assert tracker.to_dict()['human'] == steps
        assert RECORDS[2]['value'] == 12


class TestPrisma:
    def test_items_name_layers_models_licenses_and_checks(self, tracker):
        log_human(tracker, 'S01', 'duration', 'correct', 12, 14, coder='MS')
        items = prisma_items(tracker, RECORDS)
        assert set(items) == {'item 9', 'item 11'}
        assert 'gemma3:4b' in items['item 9'] and 'local_only' in items['item 9']
        assert '1 step' in items['item 9'] and 'corrected' in items['item 9']
        assert 'pubextractor' in items['item 11']
        assert not any(word in ' '.join(items.values()).lower().split()
                       for word in ('we', 'our', 'you'))

    def test_no_records_leave_no_dangling_comma(self):
        item = prisma_items(None, [])['item 9']
        assert 'from 0 papers: 0 values.' in item
        assert ', .' not in item
