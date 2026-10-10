from collections import namedtuple
from pathlib import Path

import pytest

from pubextractor import MODEL_SIZES, VISION_MODELS, ollama_models_dir, providers, storage_check

Usage = namedtuple('Usage', 'total used free')
GB = 1024 ** 3


def disk(free, total=500, seen=None):
    def usage(path):
        if seen is not None:
            seen.append(Path(path))
        return Usage(total * GB, (total - free) * GB, free * GB)
    return usage


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(providers, 'ollama_models', lambda base_url, timeout=1.5: None)


class TestModelsDir:
    def test_variable_wins(self, tmp_path):
        assert ollama_models_dir({'OLLAMA_MODELS': str(tmp_path)}) == tmp_path

    def test_default_sits_under_the_home_folder(self):
        assert ollama_models_dir({}) == Path.home() / '.ollama' / 'models'

    def test_empty_variable_falls_back(self):
        assert ollama_models_dir({'OLLAMA_MODELS': ''}) == Path.home() / '.ollama' / 'models'


class TestTables:
    def test_sizes_and_vision_set(self):
        assert MODEL_SIZES['gemma3:4b'] == 3.3
        assert 'gemma3:4b' in VISION_MODELS
        assert VISION_MODELS <= set(MODEL_SIZES)
        assert not {'tinyllama', 'gemma3:1b', 'llama3.1:8b'} & VISION_MODELS
        assert not any('qwen' in name for name in MODEL_SIZES)


class TestStorageCheck:
    def test_enough_space(self, tmp_path, monkeypatch, offline):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(210.5))
        found = storage_check('gemma3:4b', path=tmp_path)
        assert (found['free_gb'], found['total_gb'], found['needed_gb']) == (210.5, 500.0, 3.3)
        assert found['enough'] is True and found['vision'] is True
        assert found['installed'] is None
        assert 'gemma3:4b needs about 3.3 GB; 210.5 GB free on ' in found['message']
        assert found['message'].endswith('enough.')

    def test_not_enough_space_counts_the_margin(self, tmp_path, monkeypatch, offline):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(4.0))
        found = storage_check('gemma3:4b', path=tmp_path)
        assert found['enough'] is False
        assert found['message'] == 'not enough space: free 4.0 GB, need about 3.3 GB'

    def test_exact_margin_is_enough(self, tmp_path, monkeypatch, offline):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(4.3))
        assert storage_check('gemma3:4b', path=tmp_path)['enough'] is True

    def test_unknown_model(self, tmp_path, monkeypatch, offline):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(50))
        found = storage_check('mystery:7b', path=tmp_path)
        assert found['needed_gb'] is None and found['enough'] is None
        assert found['vision'] is False
        assert 'size unknown' in found['message']

    def test_missing_folder_checks_the_nearest_parent(self, tmp_path, monkeypatch, offline):
        seen = []
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(50, seen=seen))
        found = storage_check('tinyllama', path=tmp_path / 'absent' / 'models')
        assert seen == [tmp_path] and Path(found['path']) == tmp_path

    def test_missing_drive_gives_a_message_not_an_error(self, monkeypatch, offline):
        monkeypatch.setattr(providers.Path, 'exists', lambda self: False)
        found = storage_check('tinyllama', path='Z:/nope/x')
        assert (found['free_gb'], found['enough']) == (None, None)
        assert found['message'] == f"model folder drive not found: {Path('Z:/nope/x')}"

    def test_default_path_follows_the_environment(self, tmp_path, monkeypatch, offline):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(50))
        found = storage_check('tinyllama', environ={'OLLAMA_MODELS': str(tmp_path)})
        assert Path(found['path']) == tmp_path

    def test_installed_model_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(2))
        monkeypatch.setattr(providers, 'ollama_models',
                            lambda base_url, timeout=1.5: ['gemma3:4b', 'tinyllama:latest'])
        assert storage_check('gemma3:4b', path=tmp_path)['installed'] is True
        found = storage_check('tinyllama', path=tmp_path)
        assert found['installed'] is True and 'already installed' in found['message']
        assert storage_check('llava:7b', path=tmp_path)['installed'] is False

    def test_remote_endpoint_is_never_queried(self, tmp_path, monkeypatch):
        monkeypatch.setattr(providers.shutil, 'disk_usage', disk(50))
        monkeypatch.setattr(providers, 'ollama_models',
                            lambda base_url, timeout=1.5: pytest.fail('queried'))
        found = storage_check('gemma3:4b', path=tmp_path, base_url='http://example.com:11434')
        assert found['installed'] is None
