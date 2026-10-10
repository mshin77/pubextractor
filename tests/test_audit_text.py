import os
import re

import pytest

from pubextractor import fulltext
from pubextractor.evidence import find_keyword_spans, search_document
from pubextractor.fulltext import (
    detect_page_labels,
    detect_sections,
    extract_fulltext_evidence,
    read_pdf,
)


def page(number, text):
    return {'page': number, 'text': text}


def test_single_stray_number_gives_no_labels():
    pages = [page(1, 'body\n2025'), page(2, 'body'), page(3, 'body')]
    assert [p['printed_page'] for p in detect_page_labels(pages)] == [None] * 3


def test_offset_needs_two_pages():
    pages = [page(1, 'body\n10'), page(2, 'body')]
    assert [p['printed_page'] for p in detect_page_labels(pages)] == [None, None]


def test_offset_needs_majority_of_candidates():
    pages = [page(1, 'a\n5'), page(2, 'a\n6'), page(3, 'a\n40'), page(4, 'a\n90')]
    assert [p['printed_page'] for p in detect_page_labels(pages)] == [None] * 4


@pytest.mark.parametrize('line', ['Participants were 45 students.',
                                  'Results were mixed.',
                                  'Participants were recruited from three',
                                  'Results were mixed across the',
                                  'Analysis of variance showed',
                                  'Procedures were as follows',
                                  'Method was described earlier',
                                  'Analysis of variance showed a large effect on the outcome'])
def test_sentences_are_not_headings(line):
    assert detect_sections([page(1, line)])[0]['section'] == ''


@pytest.mark.parametrize('line,name', [('2. Method', 'Method'), ('Results', 'Results'),
                                       ('Materials and Methods', 'Materials and Methods'),
                                       ('Methods', 'Methods'), ('Methodology and Design', 'Methodology and Design'),
                                       ('2.1 Participants', 'Participants'),
                                       ('RESULTS', 'Results'), ('3. Results', 'Results'),
                                       ('Discussion and Conclusions', 'Discussion and Conclusions'),
                                       ('DISCUSSION AND CONCLUSIONS', 'Discussion and Conclusions'),
                                       ('Data Analysis', 'Data Analysis')])
def test_short_lines_are_headings(line, name):
    assert detect_sections([page(1, f'{line}\nbody text')])[0]['section'] == name


def test_section_carries_across_pages_and_marks_positions():
    out = detect_sections([page(1, 'Method\nstuff'), page(2, 'more\nResults\nfound')])
    assert out[1]['section_start'] == 'Method'
    assert out[1]['section'] == 'Results'
    assert out[1]['section_marks'] == [(5, 'Results')]


def test_evidence_above_mid_page_heading_keeps_earlier_section(monkeypatch):
    text = 'Method\nParticipants had autism.\nResults\nParticipants gained.'
    monkeypatch.setattr(fulltext, 'read_pdf', lambda path, max_pages=None: [page(1, text)])
    found = extract_fulltext_evidence('x.pdf', {'c': {'pattern': 'autism|gained'}})['c']
    assert [s['section'] for s in found] == ['Method', 'Results']


def test_read_pdf_rejects_nonpositive_max_pages():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            read_pdf('x.pdf', max_pages=bad)


def test_abbreviations_do_not_split_sentences(monkeypatch):
    text = 'Smith et al. reported autism in e.g. children, i.e. young. See Fig. 2 vs. 3. Done.'
    monkeypatch.setattr(fulltext, 'read_pdf', lambda path, max_pages=None: [page(1, text)])
    found = extract_fulltext_evidence('x.pdf', {'c': {'pattern': 'autism'}})['c']
    assert found[0]['text'] == 'Smith et al. reported autism in e.g. children, i.e. young.'


def test_pattern_errors_name_the_entry(monkeypatch):
    monkeypatch.setattr(fulltext, 'read_pdf', lambda path, max_pages=None: [page(1, 'x')])
    with pytest.raises(ValueError, match='crit'):
        extract_fulltext_evidence('x.pdf', {'crit': {'description': 'd'}})
    with pytest.raises(ValueError, match=re.escape('(unclosed')):
        extract_fulltext_evidence('x.pdf', {'crit': {'pattern': '(unclosed'}})


@pytest.mark.parametrize('bad', ['', '   ', None])
def test_blank_keyword_rejected(bad):
    with pytest.raises(ValueError):
        find_keyword_spans('some text', bad)


def test_keyword_spans_still_located():
    spans = find_keyword_spans('one two three four five', 'three', window=1)
    assert spans[0]['text'] == 'two three four'


def test_search_document_keyword_on_labeled_pages():
    pages = detect_sections([page(1, 'Method\nautism here')])
    assert search_document(pages, ['autism'])['autism'][0]['section'] == 'Method'


def test_highlight_save_is_atomic(tmp_path, monkeypatch):
    openpyxl = pytest.importorskip('openpyxl')
    from pubextractor.highlight import highlight_evidence
    path = tmp_path / 'a.xlsx'
    book = openpyxl.Workbook()
    book.active.append(['title'])
    book.active.append(['single case'])
    book.save(path)
    calls = []
    real = os.replace
    monkeypatch.setattr(os, 'replace', lambda a, b: (calls.append((str(a), str(b))), real(a, b))[1])
    assert highlight_evidence(path, {'c': 'single case'}) == 1
    assert calls and calls[0][1] == str(path)
    assert os.path.dirname(calls[0][0]) == str(tmp_path)
    assert os.listdir(tmp_path) == ['a.xlsx']
