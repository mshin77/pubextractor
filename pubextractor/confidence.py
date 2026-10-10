import re

NEGATIVE_MARKERS = (
    ('not found', 0.40),
    ('not mentioned', 0.30),
    ('not specified', 0.30),
    ('unclear', 0.20),
    ('uncertain', 0.20),
)


CHAR_MAP = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-',
})
QUOTE = re.compile(r'"([^"]{10,})"|\u201c([^\u201d]{10,})\u201d')
ELLIPSIS = re.compile(r'\.{3}|\u2026')
HYPHEN_BREAK = re.compile(r'-\s*\n\s*')


def _squash(text):
    return ' '.join(text.translate(CHAR_MAP).lower().split())


def _variants(text):
    return {_squash(HYPHEN_BREAK.sub(joiner, text)) for joiner in ('-', '')}


def _in_order(segments, haystack):
    position = 0
    for segment in segments:
        position = haystack.find(segment, position) + 1 or -1
        if position < 0:
            return False
        position += len(segment) - 1
    return True


def quote_found(quote, passages):
    segments = [segment for segment in map(_squash, ELLIPSIS.split(quote)) if len(segment) >= 10]
    return bool(segments) and any(_in_order(segments, haystack)
                                  for passage in passages for haystack in _variants(passage))


def score_answer(answer, passages=()):
    """Score how well an extracted answer is grounded in its source passages.

    A heuristic for triage, not a probability. Not comparable to the calibrated
    probability from a classifier or a language model's self-reported confidence.

    Args:
        answer: The extracted answer text.
        passages: Source passages the answer should be grounded in.

    Returns:
        float: Grounding score from 0.0 to 1.0.
    """
    if not answer or not isinstance(answer, str):
        return 0.0
    if answer.startswith('ERROR'):
        return 0.0

    lowered = answer.lower()
    score = 0.5

    for marker, penalty in NEGATIVE_MARKERS:
        if marker in lowered:
            score -= penalty

    quoted = [straight or curly for straight, curly in QUOTE.findall(answer)]
    unfound = [q for q in quoted if passages and not quote_found(q, passages)]
    score += min(0.2, (len(quoted) - len(unfound)) * 0.1)
    score -= 0.4 * bool(unfound)

    if re.search(r'passage\s+\d+', lowered):
        score += 0.1
    if re.search(r'(p\.|page)\s*\d+', lowered):
        score += 0.1
    if len(answer) > 50:
        score += 0.05
    if len(answer) > 150:
        score += 0.05
    if re.search(r'\b\d+\b', answer):
        score += 0.05

    if passages:
        answer_words = set(lowered.split())
        passage_words = set(' '.join(passages).lower().split())
        overlap = len(answer_words & passage_words)
        if overlap > 5:
            score += min(0.15, overlap * 0.01)

    return max(0.0, min(1.0, score))


def interpret_confidence(score):
    """Band label for a grounding score."""
    if score is None:
        return None
    if score >= 0.8:
        return 'high'
    if score >= 0.6:
        return 'moderate'
    if score >= 0.4:
        return 'low'
    return 'very low'


def needs_escalation(score, threshold=0.7):
    """Whether an extraction should go to a second reviewer.

    Returning True is a request for human attention, not a verdict on the answer.
    """
    return score is None or score < threshold


def score_extractions(extractions, passages_list=None, answer_key='answer',
                      threshold=0.7):
    """Score a list of extractions and flag the ones needing a second reviewer.

    Returns:
        list: each extraction with ``confidence``, ``confidence_band``, and
        ``escalate`` added.
    """
    passages_list = passages_list or [()] * len(extractions)
    if len(passages_list) != len(extractions):
        raise ValueError(f"passages_list length {len(passages_list)} differs from "
                         f"extractions length {len(extractions)}")
    scored = []
    for extraction, passages in zip(extractions, passages_list):
        score = score_answer(extraction.get(answer_key, ''), passages)
        scored.append({
            **extraction,
            'confidence': round(score, 4),
            'confidence_band': interpret_confidence(score),
            'escalate': needs_escalation(score, threshold),
        })
    return scored


def confidence_report(scored, threshold=0.7):
    """Counts by band and how many extractions need a second reviewer."""
    if not scored:
        return {'total': 0, 'mean_confidence': None, 'escalate': 0, 'bands': {}}
    scores = [item['confidence'] for item in scored]
    bands = {}
    for item in scored:
        bands[item['confidence_band']] = bands.get(item['confidence_band'], 0) + 1
    return {
        'total': len(scored),
        'mean_confidence': round(sum(scores) / len(scores), 4),
        'escalate': sum(1 for item in scored if needs_escalation(item['confidence'], threshold)),
        'threshold': threshold,
        'bands': bands,
    }
