from urllib import request

import pytest

from pubextractor import providers
from pubextractor.licensing import redact_endpoint


def _routed_host(url):
    handler = providers._LocalDirectProxy({'http': 'http://proxy.example:3128'})
    sent = request.Request(url)
    handler.proxy_open(sent, 'http://proxy.example:3128', 'http')
    return sent.host


def test_local_endpoint_bypasses_environment_proxy(monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.example:3128')
    assert _routed_host('http://localhost:11434/v1') == 'localhost:11434'
    assert _routed_host('http://127.0.0.1:11434/v1') == '127.0.0.1:11434'


def test_remote_endpoint_keeps_environment_proxy():
    assert _routed_host('http://api.example.com/v1') == 'proxy.example:3128'


def test_opener_has_no_plain_proxy_handler():
    assert not any(type(h) is request.ProxyHandler for h in providers._OPENER.handlers)


@pytest.mark.parametrize('url', [
    'https://h.com/v1/key=abcd',
    'https://h.com/v1/token=abcd/chat',
    'https://h.com/v1/secret=abcd',
    'https://h.com/v1/sk-abc123def456ghi789jkl012',
    'https://h.com/v1/QWxhZGRpbjpvcGVuIHNlc2FtZQ0123456789',
    'https://h.com/v1/0123456789abcdef0123456789abcdef',
])
def test_redact_endpoint_masks_credential_segments(url):
    shown = redact_endpoint(url)
    assert shown.startswith('https://h.com/v1/') and '***' in shown
    assert not any(s in shown for s in ('abcd', 'abc123', 'QWxh', '0123456789'))


@pytest.mark.parametrize('path', ['/v1', '/v1beta/openai', '/api', '/v1/chat/completions'])
def test_redact_endpoint_keeps_normal_paths(path):
    assert redact_endpoint(f"https://h.com{path}") == f"https://h.com{path}"
