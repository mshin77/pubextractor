import re
from bisect import bisect_left, bisect_right

from .fulltext import format_anchor


def _bounded(keyword):
    """Escape a keyword, anchoring word boundaries only where they can match.

    A trailing boundary after a non-word character such as ``c++`` never matches,
    so the boundary is applied per end according to the keyword's own characters.
    """
    boundary = r'\b'
    escaped = re.escape(keyword)
    left = boundary if keyword[:1].isalnum() or keyword[:1] == '_' else ''
    right = boundary if keyword[-1:].isalnum() or keyword[-1:] == '_' else ''
    return left + escaped + right


def find_keyword_spans(text, keyword, window=5, case_sensitive=False, max_matches=None):
    """Find a keyword in text with surrounding context.

    Each span carries the offset of the match inside its own text, so an interface
    can highlight the term itself rather than re-searching the context string.

    Args:
        text: Text to search.
        keyword: Word or phrase to find. Treated as a literal, not a pattern.
        window: Context words either side. An int applies both sides; a
            ``(before, after)`` pair sets them separately.
        case_sensitive: Match case exactly.
        max_matches: Stop after this many matches.

    Returns:
        list: ``{'text', 'matched_term', 'match_start', 'match_end',
        'context_start', 'context_end'}`` per hit. The context bounds are
        absolute offsets into ``text``, so a caller can tell two hits that share
        context from two that stand apart.
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError(f"keyword must be a non-blank string, got {keyword!r}")
    before, after = (window, window) if isinstance(window, int) else window
    pattern = re.compile(_bounded(keyword), 0 if case_sensitive else re.IGNORECASE)

    tokens = [(m.start(), m.end()) for m in re.finditer(r'\S+', text)]
    starts, ends = [t[0] for t in tokens], [t[1] for t in tokens]
    spans = []
    for hit in pattern.finditer(text):
        first = bisect_right(ends, hit.start())
        last = bisect_left(starts, hit.end()) - 1
        covered = list(range(first, last + 1))
        if not covered:
            continue
        lo = max(0, covered[0] - before)
        hi = min(len(tokens) - 1, covered[-1] + after)
        context = text[tokens[lo][0]:tokens[hi][1]]
        spans.append({
            'text': context,
            'matched_term': hit.group(),
            'match_start': hit.start() - tokens[lo][0],
            'match_end': hit.end() - tokens[lo][0],
            'context_start': tokens[lo][0],
            'context_end': tokens[hi][1],
        })
        if max_matches and len(spans) >= max_matches:
            break
    return spans


def search_document(pages, keywords, window=5, case_sensitive=False, max_per_page=None):
    """Search a parsed document for keywords, anchored to printed pages.

    Pages are the output of ``detect_sections(detect_page_labels(read_pdf(path)))``,
    so every hit carries the page a reader can turn to and the section it sits in.

    Returns:
        dict: ``{keyword: [span, ...]}`` with ``page``, ``pdf_page``,
        ``printed_page``, ``section``, and ``anchor`` added to each span.
    """
    results = {keyword: [] for keyword in keywords}
    for page in pages:
        for keyword in keywords:
            for span in find_keyword_spans(page['text'].replace('\n', ' '), keyword,
                                           window, case_sensitive, max_per_page):
                located = {
                    **span,
                    'page': page.get('printed_page') or page['page'],
                    'pdf_page': page['page'],
                    'printed_page': page.get('printed_page'),
                    'section': page.get('section', ''),
                }
                located['anchor'] = format_anchor(located)
                results[keyword].append(located)
    return results


def highlight_markdown(span):
    """Render a span with the matched term in markdown bold, for notebooks and sheets."""
    text, start, end = span['text'], span['match_start'], span['match_end']
    return f"{text[:start]}**{text[start:end]}**{text[end:]}"


def format_evidence(spans, separator=' | ', max_spans=3, with_anchor=True):
    """Join spans into one evidence string for a report column."""
    if not spans:
        return ''
    parts = []
    for span in spans[:max_spans] if max_spans else spans:
        rendered = highlight_markdown(span)
        parts.append(f"{span['anchor']}: {rendered}" if with_anchor and span.get('anchor')
                     else rendered)
    return separator.join(parts)
