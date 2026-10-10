import inspect
import logging

import numpy as np

from .confidence import interpret_confidence, needs_escalation, score_answer
from .fulltext import format_anchor
from .licensing import check_llm_use, is_local_endpoint, redact_endpoint

logger = logging.getLogger(__name__)

UNCHECKED = 'unchecked'

PROMPT_TEMPLATE = """Answer the question using only the passages below.

Question: {query}

Passages:
{passages}

Quote the sentence the answer rests on and give its passage number. If the
passages do not answer the question, reply exactly: not found."""


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping character chunks, breaking at whitespace."""
    if overlap < 0 or chunk_size <= overlap:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(' ', start, end)
            # breaking at a space that sits inside the overlap would put the next
            # chunk behind this one, and the loop would never reach the end
            if space > start + overlap:
                end = space
        body = text[start:end].strip()
        if body:
            chunks.append({'index': len(chunks), 'start': start, 'text': body})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages, chunk_size=1000, overlap=200):
    """Chunk each page separately, so every chunk keeps the page it came from.

    Takes the output of ``read_pdf``, optionally passed through
    ``detect_page_labels`` and ``detect_sections``, or the blocks of
    ``read_docx``, which carry paragraph and table numbers instead of pages.
    """
    if not isinstance(pages, (list, tuple)) or not all(isinstance(p, dict) for p in pages):
        raise ValueError("pages must be a list of dicts with a 'text' key, as read_pdf returns")
    chunks = [
        {**chunk, 'pdf_page': page.get('page'), 'printed_page': page.get('printed_page'),
         'paragraph': page.get('paragraph'), 'table': page.get('table'),
         'section': page.get('section', '')}
        for page in pages
        for chunk in chunk_text(page.get('text') or '', chunk_size, overlap)
    ]
    return [{**chunk, 'index': i} for i, chunk in enumerate(chunks)]


def _check_permission(permission, endpoint):
    if permission == UNCHECKED:
        logger.warning("Permission marked unchecked; nothing checks whether this text may "
                       "be sent to the model.")
        return None
    if not isinstance(permission, dict):
        raise ValueError("permission is required: pass check_llm_use(...) for this paper, "
                         "or 'unchecked' to skip the check knowingly")
    missing = {'verdict', 'allowed', 'reason', 'inputs'} - set(permission)
    if missing:
        raise ValueError(f"permission lacks {sorted(missing)}; pass the result of check_llm_use")
    names = set(inspect.signature(check_llm_use).parameters)
    inputs = permission['inputs']
    if not isinstance(inputs, dict) or set(inputs) != names:
        raise ValueError("permission inputs are incomplete; pass the result of check_llm_use")
    recomputed = check_llm_use(**inputs)
    if any(permission[key] != recomputed[key] for key in ('verdict', 'allowed')) \
            or permission['allowed'] is not recomputed['allowed']:
        raise ValueError("permission does not match its inputs; pass the result of "
                         "check_llm_use unchanged")
    if not recomputed['allowed']:
        raise PermissionError(f"{recomputed['verdict']}: {recomputed['reason']}")
    processing = inputs['processing']
    if processing == 'api':
        return recomputed
    if not endpoint:
        raise PermissionError(f"Processing declared as {processing}; pass the endpoint "
                              "respond calls so it can be checked and recorded")
    if processing == 'local' and not is_local_endpoint(endpoint):
        raise PermissionError(f"Processing declared as local, but the endpoint "
                              f"{redact_endpoint(endpoint)} does not plainly name this machine "
                              "(user names, '%', and '\\' are refused)")
    return recomputed


def _anchor(chunk):
    return format_anchor(chunk) if 'pdf_page' in chunk else f"chunk {chunk['index']}"


def retrieve(query, chunks, embeddings, model, top_k=5):
    """Rank chunks against a query by cosine similarity.

    Retrieval runs against the embeddings in memory, so no vector database and no
    external service is involved.

    Returns:
        list: top chunks, each with a ``similarity`` value, highest first.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    query_vector = np.atleast_2d(np.asarray(model.encode([query]), dtype=float))
    matrix = np.asarray(embeddings, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != len(chunks) or matrix.shape[1] == 0
            or query_vector.shape != (1, matrix.shape[1])
            or not np.isfinite(matrix).all() or not np.isfinite(query_vector).all()):
        raise ValueError('embedding dimensions or values do not match the chunks and query')
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    norms[norms == 0] = 1e-12
    scores = (matrix @ query_vector.T).ravel() / norms

    order = np.argsort(-scores)[:top_k]
    return [{**chunks[i], 'similarity': float(scores[i])} for i in order]


def build_index(chunks, model, batch_size=32):
    """Embed chunks once for reuse across queries."""
    texts = [chunk['text'] for chunk in chunks]
    return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False))


def extract_with_rag(text, queries, respond, model, top_k=5, chunk_size=1000,
                     overlap=200, template=None, tracker=None, threshold=0.7,
                     permission=None, endpoint=None, paper_id=None, cache=None):
    """Answer questions about a document from retrieved passages.

    Retrieval is local; generation goes through ``respond``. Each answer carries
    its passages and a grounding score; answers below ``threshold`` are flagged.

    Args:
        text: Document text, or the page list from ``read_pdf`` so each passage
            keeps its page and section.
        queries: ``{name: query}`` or ``{name: {'query': str}}``.
        respond: Callable mapping a prompt to reply text.
        model: Sentence encoder with an ``encode`` method.
        top_k: Passages retrieved per query.
        chunk_size: Characters per chunk.
        overlap: Characters shared between neighboring chunks.
        template: Prompt template overriding the default. Must accept ``query``
            and ``passages``.
        tracker: ProvenanceTracker to record the retrieval configuration and the
            prompts actually sent.
        threshold: Grounding score below which an answer is escalated.
        permission: Result of ``check_llm_use`` for this paper, unchanged; it is
            recomputed from its inputs and refused if edited. A refusal raises
            ``PermissionError`` before any text reaches ``respond``. Pass
            ``'unchecked'`` to skip the check knowingly.
        endpoint: URL of the model that ``respond`` calls. Required unless the
            permission declares ``processing='api'``; for ``'local'`` it must
            name this machine. Only the host name is checked; whether
            ``respond`` really calls this URL remains the caller's
            responsibility.
        paper_id: Identifier recorded with the license verdict, so one tracker
            can hold many papers.
        cache: Dict reused across calls on the same document, so chunks are
            embedded once. Pass a fresh dict per document.

    Returns:
        dict: ``{name: {'answer', 'passages', 'confidence', 'confidence_band',
        'escalate', 'error'}}``; ``error`` names the exception type when the
        model call failed, else None. Each passage carries an ``anchor`` such as ``p. 576 · Method``.
    """
    checked = _check_permission(permission, endpoint)
    if not isinstance(text, (str, list, tuple)):
        raise ValueError("text must be a string or the page list from read_pdf")
    if not isinstance(queries, dict) or not all(
            isinstance(s, str) or (isinstance(s, dict) and isinstance(s.get('query'), str))
            for s in queries.values()):
        raise ValueError("queries must map names to a query string or {'query': str}")

    prompt_template = template or PROMPT_TEMPLATE
    key = (chunk_size, overlap, repr(text))
    if cache is not None and cache.get('key') == key and cache.get('model') is model:
        chunks, embeddings = cache['chunks'], cache['embeddings']
    else:
        chunks = (chunk_text(text, chunk_size, overlap) if isinstance(text, str)
                  else chunk_pages(text, chunk_size, overlap))
        if not chunks:
            raise ValueError("Document has no extractable text; a scanned PDF needs OCR first")
        chunks = [{**chunk, 'anchor': _anchor(chunk)} for chunk in chunks]
        embeddings = build_index(chunks, model)
        cache is None or cache.update(key=key, model=model, chunks=chunks, embeddings=embeddings)

    if tracker is not None:
        record = {'verdict': UNCHECKED} if checked is None else checked
        tracker.log_permission({**record,
                                'endpoint': redact_endpoint(endpoint) if endpoint else None},
                               paper_id=paper_id)
    if tracker is not None:
        tracker.log_retrieval(
            embedding_model=getattr(model, 'name_or_path', str(type(model).__name__)),
            chunk_size=chunk_size, chunk_overlap=overlap, top_k=top_k)

    results = {}
    for name, spec in queries.items():
        query = spec['query'] if isinstance(spec, dict) else spec
        passages = retrieve(query, chunks, embeddings, model, top_k)
        numbered = '\n\n'.join(
            f"[{i + 1}] ({p['anchor']}) {p['text']}" for i, p in enumerate(passages))
        prompt = prompt_template.format(query=query, passages=numbered)

        error = None
        try:
            answer = respond(prompt)
            if not isinstance(answer, str):
                raise ValueError('model reply must be text')
        except Exception as failure:
            logger.exception(f"Model call failed for {name}")
            answer, error = 'ERROR: model call failed', type(failure).__name__

        if tracker is not None:
            tracker.log_prompt(prompt, model=getattr(respond, 'model', None))

        score = score_answer(answer, [p['text'] for p in passages])
        results[name] = {
            'answer': answer,
            'passages': passages,
            'confidence': round(score, 4),
            'confidence_band': interpret_confidence(score),
            'escalate': needs_escalation(score, threshold),
            'error': error,
        }

    escalated = sum(1 for r in results.values() if r['escalate'])
    logger.info(f"{len(results)} queries answered, {escalated} flagged for review")
    return results
