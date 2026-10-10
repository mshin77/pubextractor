import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEPENDENCIES = ('numpy', 'pdfplumber', 'openpyxl', 'python-docx', 'sentence-transformers')


class ProvenanceTracker:
    """Record what a run did, so a reported result can be reconstructed.

    Values are recorded as given, not inferred. Prompts are hashed, so a claim
    about what a provider received can be checked.
    """

    def __init__(self, stage, task):
        self.stage = stage
        self.task = task
        self.metadata = {
            'stage': stage,
            'task': task,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'environment': self._environment(),
        }

    def _environment(self):
        from . import __version__

        info = {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'pubextractor': __version__,
        }
        for name in DEPENDENCIES:
            try:
                info[name] = version(name)
            except PackageNotFoundError:
                info[name] = None
        return info

    def log_model_config(self, model_name, model_type=None, checkpoint=None, **kwargs):
        self.metadata['model'] = {
            'name': model_name,
            'type': model_type,
            'checkpoint': checkpoint,
            **kwargs,
        }
        return self

    def log_random_seed(self, seed):
        self.metadata['random_seed'] = seed
        return self

    def log_data(self, path=None, n_records=None, n_labeled=None, **kwargs):
        self.metadata['data'] = {
            'path': str(path) if path else None,
            'records': n_records,
            'labeled': n_labeled,
            **kwargs,
        }
        return self

    def log_retrieval(self, embedding_model=None, chunk_size=None, chunk_overlap=None,
                      top_k=None, **kwargs):
        self.metadata['retrieval'] = {
            'embedding_model': embedding_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'top_k': top_k,
            **kwargs,
        }
        return self

    def log_threshold(self, threshold, target_recall=None, basis=None):
        self.metadata['threshold'] = {
            'value': threshold,
            'target_recall': target_recall,
            'basis': basis,
        }
        return self

    def log_permission(self, permission, paper_id=None):
        """Record the license verdict each paper was processed under."""
        self.metadata.setdefault('permissions', {})['unnamed' if paper_id is None else paper_id] = dict(permission)
        return self

    def log_criteria(self, criteria):
        """Record criterion descriptions verbatim; their wording drives semantic scores."""
        self.metadata['criteria'] = dict(criteria)
        return self

    def log_prompt(self, prompt, model=None, parameters=None):
        """Record a prompt by hash, with the model it went to and a running digest.

        The text itself is not stored, so passages from licensed papers are not
        copied into the log.
        """
        prompts = self.metadata.setdefault('prompts', {
            'parameters': parameters or {},
            'count': 0, 'digest': '', 'records': [],
        })
        prompts['count'] += 1
        prompts['digest'] = hashlib.sha256(
            (prompts['digest'] + prompt).encode('utf-8')).hexdigest()
        prompts['records'].append({
            'n': prompts['count'], 'model': model,
            'sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        })
        return self

    def add_note(self, key, value):
        self.metadata.setdefault('notes', {})[key] = value
        return self

    def to_dict(self):
        return dict(self.metadata)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding='utf-8')
        return path


def log_human(tracker, paper_id, field, action, before, after, coder=None, time=None):
    """Record one coder step (confirm, correct, role, calibrate, enter) and return it.

    With no tracker the step is returned without being recorded.
    """
    step = {'paper_id': paper_id, 'field': field, 'layer': 'human', 'action': action,
            'before': before, 'after': after, 'coder': coder,
            'time': time or datetime.now(timezone.utc).isoformat()}
    if tracker is not None:
        tracker.metadata.setdefault('human', []).append(step)
    return step


def load_provenance(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_provenances(first, second):
    """Report which recorded fields differ between two runs.

    Run and record timestamps are ignored, since they differ by construction.
    """
    left = load_provenance(first) if isinstance(first, (str, Path)) else first
    right = load_provenance(second) if isinstance(second, (str, Path)) else second

    def flatten(node, prefix=''):
        flat = {}
        for key, value in node.items():
            if key == 'created_at':
                continue
            label = f'{prefix}.{key}' if prefix else key
            if isinstance(value, dict):
                flat.update(flatten(value, label))
            elif isinstance(value, list):
                flat[label] = [{k: v for k, v in item.items() if k != 'time'}
                               if isinstance(item, dict) else item for item in value]
            else:
                flat[label] = value
        return flat

    flat_left, flat_right = flatten(left), flatten(right)
    keys = set(flat_left) | set(flat_right)
    differences = {k: {'first': flat_left.get(k), 'second': flat_right.get(k)}
                   for k in sorted(keys) if flat_left.get(k) != flat_right.get(k)}
    return {'identical': not differences, 'differences': differences}
