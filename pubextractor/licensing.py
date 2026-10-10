"""Per-paper check of whether text may go to a language model, and where it may run.

A verdict is guidance for the reviewer, not legal advice. It reflects the rules as
of RULES_DATE and covers copyright and license terms only, not privacy or data
protection. The institution's license with each publisher, and its counsel,
override any verdict. Every input is declared by the caller, so a wrong input
gives a wrong verdict; provider terms are not checked; and which country's law
applies can depend on where the reviewer and the server are. ``extract_with_rag``
checks the declared processing against the endpoint's host name, nothing more.
Treating a researcher-run cloud model (``private_cloud``) as stricter than
``local`` is a policy choice of these rules, not a legal line.
"""

import ipaddress
import re
from urllib.parse import urlparse

RULES_VERSION = '0.3'
RULES_DATE = '2026-10-04'

OPEN_LICENSES = {'cc-by', 'cc-by-sa', 'cc0', 'public-domain'}
NONCOMMERCIAL_LICENSES = {'cc-by-nc', 'cc-by-nd', 'cc-by-nc-sa', 'cc-by-nc-nd'}

EXCEPTIONS = {
    'us': 'US fair use (17 U.S.C. §107), which is decided case by case',
    'eu': ('the EU DSM Directive Art. 3 exception for research organisations, which '
           'requires lawful access and secure storage (Art. 3(2))'),
    'uk': 'the UK CDPA s29A exception for non-commercial research with lawful access',
    'other': 'local copyright law (confirm with the library)',
}

# where an exception covers only some reviewers, the others are referred onward
LIMITS = {
    'eu': ('Outside a research organisation the EU DSM Directive Art. 4 applies, and '
           'publishers may opt out of it'),
    'uk': 'The UK CDPA s29A exception covers only non-commercial research',
}

# contract terms that bar a local copy for text and data mining may not bind
OVERRIDES = {
    'eu': (' For a research organisation the term may be unenforceable against a local '
           'copy made for text and data mining (EU DSM Directive Art. 7(1)); confirm '
           'with the library.'),
    'uk': (' The term may be unenforceable against a local copy made for non-commercial '
           'text and data mining (UK CDPA s29A(5)); confirm with the library.'),
}

CLOSED_TERMS = 'no_training_no_retention'

CHOICES = {
    'content': {'abstract', 'fulltext'},
    'access': {'open_access', 'subscription', 'personal'},
    'ai_terms': {'allows', 'silent', 'forbids', 'unknown'},
    'processing': {'local', 'private_cloud', 'api'},
    'provider_terms': {'standard', CLOSED_TERMS},
    'region': set(EXCEPTIONS),
}

# a route is where the text goes; an api route counts as closed only when the
# provider neither trains on nor keeps what it receives
ALLOWED = {
    'any_llm': {'local', 'private_cloud', 'api', 'api_closed'},
    'closed_ai': {'local', 'private_cloud', 'api_closed'},
    'local_only': {'local'},
    'manual': set(),
}

# first matching rule decides; a license goes with the work, so it is read before
# how the copy was obtained
RULES = (
    (lambda c: c['content'] == 'abstract' and c['ai_terms'] == 'forbids', 'closed_ai',
     'Abstracts are short and widely indexed, so the risk is low, but they remain '
     'copyrighted and these terms forbid AI tools; use a model that neither trains on '
     'nor keeps them.'),
    (lambda c: c['content'] == 'abstract', 'any_llm',
     'Abstracts are short and widely indexed, so the risk is low; they remain '
     'copyrighted.'),
    (lambda c: c['license_type'] in OPEN_LICENSES, 'any_llm',
     'Open license permits reuse, with attribution where required.'),
    (lambda c: c['license_type'] in NONCOMMERCIAL_LICENSES, 'closed_ai',
     'Non-commercial and no-derivatives terms bind commercial use and public sharing, '
     'not private reading; as a precaution against a provider training on the text, '
     'send it only to a model that neither trains on nor keeps it.'),
    (lambda c: c['ai_terms'] == 'forbids', 'manual',
     'The terms forbid AI tools; code by hand or obtain permission.{override}'),
    (lambda c: c['ai_terms'] == 'allows', 'closed_ai',
     'The terms permit AI tools; such terms usually require one that neither trains on '
     'nor keeps the content, so confirm the exact wording.'),
    (lambda c: (c['region'] in LIMITS and not c['research_org']
                and (c['access'] == 'open_access' or c['ai_terms'] == 'silent')), 'manual',
     '{limit}; check for a reservation or ask the publisher.'),
    (lambda c: c['access'] == 'open_access', 'local_only',
     'Free to read but no recognized open license; processing on this machine relies on '
     '{exception}.'),
    (lambda c: c['access'] == 'subscription' and c['ai_terms'] == 'silent', 'local_only',
     'The license is silent on AI tools; processing on this machine relies on '
     '{exception}.'),
    (lambda c: True, 'manual',
     'Personal-use copy or unconfirmed license terms; ask the library or the publisher.'),
)


def is_local_endpoint(url):
    """True when a model endpoint names a host on this machine.

    Reads the host name only. A tunnel or port forward that ends on localhost
    still reads as local, and inside a cloud notebook such as Colab localhost is
    the cloud machine, not the reviewer's computer; declare that as
    ``processing='private_cloud'``.
    """
    if not isinstance(url, str) or any(c in '\\%' or c.isspace() or not c.isprintable()
                                       for c in url):
        return False
    try:
        parsed = urlparse(url)
        _ = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in ('http', 'https') or '@' in parsed.netloc:
        return False
    host = (parsed.hostname or '').lower().rstrip('.')
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host == 'localhost'
    mapped = getattr(address, 'ipv4_mapped', None)
    return (mapped or address).is_loopback


_SECRET_SEGMENT = re.compile(
    r'(?i)(?:key|token|secret|apikey|api_key|password)=|sk-|[A-Za-z0-9+/_-]{20,}')


def redact_endpoint(url):
    """Endpoint without user name, password, query, or fragment, for logs and messages."""
    try:
        parsed = urlparse(url if isinstance(url, str) else '')
        port = f":{parsed.port}" if parsed.port else ''
    except ValueError:
        return '<unparseable endpoint>'
    host = parsed.hostname or ''
    host = f"[{host}]" if ':' in host else host
    path = '/'.join('***' if _SECRET_SEGMENT.search(s) else s for s in parsed.path.split('/'))
    return f"{parsed.scheme}://{host}{port}{path}" if parsed.scheme else '<no scheme>'


def _normalize_license(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    code = re.sub(r'[\s_]+', '-', str(value).strip().lower())
    return re.sub(r'-\d+(\.\d+)?$', '', code) or None


def check_llm_use(content='fulltext', access='subscription', license_type=None,
                  ai_terms='unknown', processing='local', region='us',
                  provider_terms='standard', research_org=True):
    """Decide whether one paper may be sent to a language model.

    Args:
        content: ``'abstract'`` for titles and abstracts, ``'fulltext'`` otherwise.
        access: ``'open_access'``, ``'subscription'``, or ``'personal'`` (interlibrary
            loan, author-shared, or purchased copy).
        license_type: Creative Commons code such as ``'cc-by'``, whatever the
            access route, since a license goes with the work. Spellings such as ``'CC BY 4.0'`` are read as ``'cc-by'``.
        ai_terms: What the publisher license says about AI tools: ``'allows'``,
            ``'silent'``, ``'forbids'``, or ``'unknown'``.
        processing: ``'local'`` for a model on this machine, ``'private_cloud'``
            for a model the reviewer runs on rented or institutional cloud,
            including Google-run notebooks such as Colab, ``'api'`` for a
            provider's service.
        region: ``'us'``, ``'eu'``, ``'uk'``, or ``'other'``.
        provider_terms: For ``'api'``, ``'no_training_no_retention'`` when the
            provider's terms say it neither trains on nor keeps the input,
            ``'standard'`` otherwise.
        research_org: ``True`` or ``False``. For ``'eu'``, whether the reviewer
            works for a research organization or cultural heritage institution
            (DSM Art. 3); for ``'uk'``, whether the work is non-commercial
            research (s29A).

    Returns:
        dict: ``{'verdict', 'allowed', 'reason', 'rules_version', 'rules_date',
        'inputs'}``. ``verdict`` is ``'any_llm'``, ``'closed_ai'``,
        ``'local_only'``, or ``'manual'``.
    """
    inputs = {'content': content, 'access': access, 'ai_terms': ai_terms,
              'processing': processing, 'region': region, 'provider_terms': provider_terms}
    invalid = [f"{k}={v!r}" for k, v in inputs.items()
               if not isinstance(v, str) or v not in CHOICES[k]]
    if invalid:
        raise ValueError(f"Unknown value: {', '.join(invalid)}")
    if type(research_org).__name__ not in ('bool', 'bool_'):
        raise ValueError(f"research_org must be True or False, not {research_org!r}")

    inputs.update(license_type=_normalize_license(license_type),
                  research_org=bool(research_org))
    _, verdict, reason = next(rule for rule in RULES if rule[0](inputs))
    route = ('api_closed' if processing == 'api' and provider_terms == CLOSED_TERMS
             else processing)
    return {
        'verdict': verdict,
        'allowed': route in ALLOWED[verdict],
        'reason': reason.format(exception=EXCEPTIONS[region], limit=LIMITS.get(region, ''),
                                override=OVERRIDES.get(region, '')),
        'rules_version': RULES_VERSION,
        'rules_date': RULES_DATE,
        'inputs': inputs,
    }
