"""Model clients for OpenAI-compatible chat endpoints, built on the standard library."""

import base64
import json
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from urllib import error, request

from .licensing import is_local_endpoint, redact_endpoint


def _pair(line):
    key, value = line.split('=', 1)
    key = key.strip()
    key = key[len('export '):].strip() if key.startswith('export ') else key
    value = value.strip()
    quoted = re.fullmatch(r'(["\'])(.*)\1(?:\s+#.*)?', value)
    return key, quoted.group(2) if quoted else re.split(r'\s#', value, maxsplit=1)[0].strip()


def _warn_if_tracked(path):
    try:
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', path.name],
                                 cwd=path.parent, capture_output=True)
        result = subprocess.run(['git', 'check-ignore', '-q', path.name], cwd=path.parent,
                                capture_output=True)
    except OSError:
        return
    if tracked.returncode == 0:
        warnings.warn(f"{path.name} is tracked by git; run git rm --cached {path.name} and "
                      "rotate any keys it held", UserWarning, stacklevel=3)
        return
    # 1 means inside a repository and not ignored; 128 means not a repository
    if result.returncode == 1:
        warnings.warn(f"{path.name} is inside a git folder but not ignored; add it to "
                      ".gitignore so keys are never committed", UserWarning, stacklevel=3)


def load_env(path='.env', environ=None):
    """Read KEY=VALUE lines into the environment without overriding variables already set.

    Returns the names of the variables set, never their values.
    """
    environ = os.environ if environ is None else environ
    path = Path(path)
    if not path.is_file():
        return []
    _warn_if_tracked(path)
    if path.read_bytes()[:2] in (b'\xff\xfe', b'\xfe\xff'):
        raise ValueError(f"{path.name} is UTF-16; save it as UTF-8")
    lines = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines()]
    parsed = [_pair(line) for line in lines if '=' in line and not line.startswith('#')]
    pairs = dict(pair for pair in parsed if pair[1])
    loaded = {key: value for key, value in pairs.items() if key not in environ}
    environ.update(loaded)
    return sorted(loaded)


# base URL of each provider's OpenAI-compatible chat API and its key variable; checked 2026-10-04
PROVIDERS = {
    'ollama': ('http://localhost:11434/v1', None),
    'openai': ('https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai', 'GEMINI_API_KEY'),
    'anthropic': ('https://api.anthropic.com/v1', 'ANTHROPIC_API_KEY'),
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
    'huggingface': ('https://router.huggingface.co/v1', 'HF_TOKEN'),
}

MODEL_SIZES = {
    'gemma3:1b': 0.8,
    'gemma3:4b': 3.3,
    'gemma3:12b': 8.1,
    'gemma3:27b': 17,
    'llama3.2-vision:11b': 7.8,
    'llava:7b': 4.7,
    'llama3.1:8b': 4.9,
    'tinyllama': 0.6,
}
"""Approximate download size in GB of local models, as listed in the Ollama library."""

VISION_MODELS = frozenset({'gemma3:4b', 'gemma3:12b', 'gemma3:27b', 'llama3.2-vision:11b',
                           'llava:7b'})
"""Models in ``MODEL_SIZES`` that read images; the rest take text only."""

SPACE_MARGIN_GB = 1.0


def ollama_models_dir(environ=None):
    """Folder where Ollama keeps models: OLLAMA_MODELS, else ``~/.ollama/models``."""
    environ = os.environ if environ is None else environ
    custom = environ.get('OLLAMA_MODELS')
    return Path(custom) if custom else Path.home() / '.ollama' / 'models'


def ollama_models(base_url, timeout=1.5):
    """Model names an Ollama server lists at /api/tags, or None when it is unreachable."""
    root = re.sub(r'/v1/?$', '', (base_url or '').rstrip('/'))
    try:
        with _OPENER.open(f"{root}/api/tags", timeout=timeout) as reply:
            return [m['name'] for m in json.loads(reply.read().decode('utf-8'))['models']]
    except Exception:
        return None


def _is_installed(model, base_url):
    names = ollama_models(base_url) if is_local_endpoint(base_url) else None
    return None if names is None else bool({model, f"{model}:latest"} & set(names))


def storage_check(model='gemma3:4b', path=None, environ=None, base_url='http://localhost:11434'):
    """Free disk space against the download size of a local model.

    ``path`` defaults to the Ollama models folder; the nearest existing parent is measured,
    and a missing drive gives ``free_gb`` and ``enough`` None.
    ``installed`` is None unless ``base_url`` is a local server that answers.
    """
    target = Path(path) if path else ollama_models_dir(environ)
    found = next((p for p in (target, *target.parents) if p.exists()), None)
    needed = MODEL_SIZES.get(model)
    if found is None:
        return {'path': str(target), 'free_gb': None, 'total_gb': None, 'needed_gb': needed,
                'installed': None, 'enough': None, 'vision': model in VISION_MODELS,
                'message': f"model folder drive not found: {target}"}
    target = found
    usage = shutil.disk_usage(target)
    free, total = round(usage.free / 1024 ** 3, 1), round(usage.total / 1024 ** 3, 1)
    enough = None if needed is None else free >= needed + SPACE_MARGIN_GB
    installed = _is_installed(model, base_url)
    where = f"{free} GB free on {target.anchor or target}"
    message = (f"{model} already installed; {where}." if installed else
               f"{model} size unknown; {where}." if needed is None else
               f"{model} needs about {needed} GB; {where} — enough." if enough else
               f"not enough space: free {free} GB, need about {needed} GB")
    return {'path': str(target), 'free_gb': free, 'total_gb': total, 'needed_gb': needed,
            'installed': installed, 'enough': enough, 'vision': model in VISION_MODELS,
            'message': message}


IMAGE_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
               '.gif': 'image/gif', '.webp': 'image/webp'}


def _data_url(image):
    if isinstance(image, (bytes, bytearray)):
        return 'data:image/png;base64,' + base64.b64encode(bytes(image)).decode('ascii')
    path = Path(image)
    kind = IMAGE_TYPES.get(path.suffix.lower(), 'image/png')
    return f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode('ascii')


class _RefuseRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"{redact_endpoint(req.full_url)} redirected to "
                           f"{redact_endpoint(newurl)}; redirect refused so the key stays put")


class _LocalDirectProxy(request.ProxyHandler):
    def proxy_open(self, req, proxy, type):
        if is_local_endpoint(req.full_url):
            return None
        return super().proxy_open(req, proxy, type)


_OPENER = request.build_opener(_RefuseRedirect, _LocalDirectProxy)


def _post(url, body, headers, timeout):
    sent = request.Request(url, data=json.dumps(body).encode('utf-8'),
                           headers={**headers, 'Content-Type': 'application/json'},
                           method='POST')
    try:
        with _OPENER.open(sent, timeout=timeout) as reply:
            return json.loads(reply.read().decode('utf-8'))
    except error.HTTPError as failure:
        raise RuntimeError(f"{redact_endpoint(url)} returned {failure.code}") from None
    except error.URLError as failure:
        raise RuntimeError(f"{redact_endpoint(url)} unreachable: {failure.reason}") from None
    except (TimeoutError, OSError) as failure:
        raise RuntimeError(f"{redact_endpoint(url)} unreachable: {failure}") from None
    except json.JSONDecodeError:
        raise RuntimeError(f"{redact_endpoint(url)} gave no reply text") from None


def make_responder(provider, model, base_url=None, key_env=None, environ=None,
                   timeout=120, post=None):
    """Build ``respond(prompt, images=None)`` for an OpenAI-compatible chat API.

    Returns ``(respond, endpoint)``; pass ``endpoint`` to ``extract_with_rag`` so
    the license check sees where the text goes. An endpoint not on this machine
    is an API, whatever the code runs on.
    """
    environ = os.environ if environ is None else environ
    default_base, default_key = PROVIDERS.get(provider, (None, None))
    base = (base_url or default_base or '').rstrip('/')
    if not base:
        raise ValueError(f"Unknown provider {provider!r}; pass base_url")
    if base_url and key_env is None and default_key:
        raise ValueError(f"{provider!r} needs a key but base_url is custom; "
                         f"pass key_env={default_key!r} to send it, or another provider name")
    key_name = key_env if key_env is not None else default_key
    key = environ.get(key_name) if key_name else None
    if key and not is_local_endpoint(base) and not base.lower().startswith('https://'):
        raise ValueError(f"Refusing to send {key_name} over plain http to {redact_endpoint(base)}")
    if key_name and not key and not is_local_endpoint(base):
        raise ValueError(f"{key_name} is not set; add it to the environment or a .env file")
    headers = {'Authorization': f"Bearer {key}"} if key else {}
    transport = post or _post

    def respond(prompt, images=None):
        content = (prompt if not images else
                   [{'type': 'text', 'text': prompt}]
                   + [{'type': 'image_url', 'image_url': {'url': _data_url(image)}}
                      for image in images])
        body = {'model': model, 'messages': [{'role': 'user', 'content': content}]}
        reply = transport(f"{base}/chat/completions", body, headers, timeout)
        try:
            content = reply['choices'][0]['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(f"{redact_endpoint(base)} gave no reply text")
            return content
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"{redact_endpoint(base)} gave no reply text") from None

    respond.model, respond.provider = model, provider
    return respond, base
