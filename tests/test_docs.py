from pathlib import Path

import pubextractor

API = Path(__file__).parents[1] / 'docs_src' / 'api.md'


def test_api_reference_lists_every_export():
    text = API.read_text(encoding='utf-8')
    assert [name for name in pubextractor.__all__ if f'`{name}`' not in text] == []


def test_api_reference_lists_tracker_loggers():
    text = API.read_text(encoding='utf-8')
    loggers = ('log_model_config', 'log_random_seed', 'log_data', 'log_threshold',
               'log_criteria', 'add_note')
    assert [name for name in loggers if f'ProvenanceTracker.{name}' not in text] == []
