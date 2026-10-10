import json

import numpy as np
import pytest

from pubextractor.codebook import load_codebook
from pubextractor.fulltext import detect_sections
from pubextractor.layers import extract_values
from pubextractor.licensing import check_llm_use
from pubextractor.output import COLUMNS

PAGES = [
    {'page': 1, 'printed_page': 101, 'section': 'Method',
     'text': 'Participants were middle school students. In total, N = 48 took part.'},
    {'page': 2, 'printed_page': 102, 'section': 'Results',
     'text': 'Students wrote journals and were observed weekly. Scores rose, t(46) = 2.10.'},
]

BOOK = load_codebook({
    'grade_band': {'type': 'category', 'values': ['elementary', 'middle', 'high'],
                   'terms': {'middle': ['middle school'], 'high': ['high school']}},
    'data_collection': {'type': 'multi_category', 'values': ['interview', 'writing', 'observation'],
                        'terms': {'writing': ['journal*'], 'observation': ['observ*'],
                                  'interview': ['interview*']}},
    'sample_size': {'type': 'integer', 'stat': 'n', 'range': [1, 100000]},
    'setting': {'type': 'text'},
})


def by_field(records):
    return {name: [r for r in records if r['field'] == name] for name in BOOK}


class TestRuleLayers:
    def test_category_found_by_terms_with_location(self):
        record = by_field(extract_values(PAGES, BOOK, paper_id='S01'))['grade_band'][0]
        assert (record['value'], record['layer'], record['status']) == ('middle', 'regex', 'extracted')
        assert (record['pdf_page'], record['printed_page'], record['section']) == (1, 101, 'Method')
        assert 'middle school' in record['quote'] and record['paper_id'] == 'S01'

    def test_multi_category_gives_one_record_per_value(self):
        records = by_field(extract_values(PAGES, BOOK))['data_collection']
        assert sorted(r['value'] for r in records) == ['observation', 'writing']
        assert all(r['status'] == 'extracted' for r in records)

    def test_number_from_reported_statistic(self):
        record = by_field(extract_values(PAGES, BOOK))['sample_size'][0]
        assert (record['value'], record['layer'], record['pdf_page']) == (48, 'regex', 1)
        assert isinstance(record['value'], int)

    def test_text_field_without_model_is_not_found(self):
        record = by_field(extract_values(PAGES, BOOK))['setting'][0]
        assert (record['value'], record['layer'], record['status']) == (None, 'none', 'not_found')

    def test_several_candidates_need_review(self):
        pages = [{'page': 1, 'text': 'Middle school and high school students took part.'}]
        record = by_field(extract_values(pages, BOOK))['grade_band'][0]
        assert record['status'] == 'review' and record['candidates'] == 2

    def test_out_of_range_numbers_are_ignored(self):
        book = load_codebook({'sample_size': {'type': 'integer', 'stat': 'n', 'range': [10, 500]}})
        pages = [{'page': 1, 'text': 'Each class had n = 4 tutors; N = 120 students.'}]
        record = extract_values(pages, book)[0]
        assert record['value'] == 120 and record['status'] == 'extracted'

    def test_every_record_has_the_shared_keys(self):
        keys = set(COLUMNS)
        assert all(set(r) == keys for r in extract_values(PAGES, BOOK))

    def test_match_below_mid_page_heading_takes_its_section(self):
        text = 'Introduction\nMiddle school pupils were studied.\nMethod\nTotal N = 48 took part.\n'
        pages = detect_sections([{'page': 1, 'text': text}])
        records = by_field(extract_values(pages, BOOK))
        assert pages[0]['section'] == 'Method'
        assert records['grade_band'][0]['section'] == 'Introduction'
        assert records['sample_size'][0]['section'] == 'Method'


def test_leading_space_does_not_shift_the_section():
    pages = [{'page': 1, 'text': ' ' * 30 + 'N = 40 took part.', 'section': 'Method',
              'section_marks': [(0, 'Intro'), (20, 'Method')]}]
    book = load_codebook({'n': {'type': 'integer', 'stat': 'n'}})
    assert extract_values(pages, book)[0]['section'] == 'Method'


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[float('setting' in t or 'school' in t), 1.0] for t in texts])


def responder(reply):
    sent = []

    def respond(prompt, images=None):
        sent.append(prompt)
        return reply if isinstance(reply, str) else json.dumps(reply)
    respond.sent = sent
    return respond


SETTING_PAGES = [{'page': 3, 'printed_page': 103, 'section': 'Method',
                  'text': 'The study took place in two urban school settings.'}]
TEXT_BOOK = load_codebook({'setting': {'type': 'text', 'query': 'Where did the study take place?'},
                           'region': {'type': 'category', 'values': ['urban', 'rural']}})
LOCAL = dict(permission=check_llm_use(access='subscription', ai_terms='silent'),
             endpoint='http://localhost:11434/v1')


class TestPromptLayer:
    def test_grounded_text_answer_is_extracted_with_document_location(self):
        respond = responder({'value': 'two urban schools',
                             'quote': 'The study took place in two urban school settings.',
                             'passage': 1})
        record = extract_values(SETTING_PAGES, {'setting': TEXT_BOOK['setting']},
                                respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert (record['value'], record['layer'], record['status']) == ('two urban schools', 'prompt', 'extracted')
        assert (record['pdf_page'], record['printed_page']) == (3, 103)

    def test_choice_outside_the_list_needs_review(self):
        respond = responder({'value': 'suburban', 'quote': 'urban school settings', 'passage': 1})
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                                respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert record['value'] is None and record['status'] == 'review'

    def test_prompt_lists_the_allowed_values(self):
        respond = responder({'value': 'urban', 'quote': 'two urban school settings', 'passage': 1})
        extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                       respond=respond, encoder=Encoder(), **LOCAL)
        assert '["urban", "rural"]' in respond.sent[0]

    def test_quote_not_in_passage_needs_review(self):
        respond = responder({'value': 'urban', 'quote': 'invented sentence', 'passage': 1})
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                                respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert record['value'] == 'urban' and record['status'] == 'review'

    def test_unparseable_reply_needs_review(self):
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                                respond=responder('not json'), encoder=Encoder(), **LOCAL)[0]
        assert record['value'] is None and record['status'] == 'review'

    def test_license_refusal_sends_nothing(self):
        respond = responder({'value': 'urban', 'quote': '', 'passage': 1})
        refused = check_llm_use(access='subscription', ai_terms='forbids')
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']}, respond=respond,
                                encoder=Encoder(), permission=refused,
                                endpoint='http://localhost:11434/v1')[0]
        assert record['status'] == 'refused' and 'forbid' in record['quote']
        assert respond.sent == []

    def test_rules_win_before_the_model(self):
        respond = responder({'value': 'rural', 'quote': '', 'passage': 1})
        book = load_codebook({'region': {'type': 'category', 'values': ['urban', 'rural'],
                                         'terms': {'urban': ['urban']}}})
        record = extract_values(SETTING_PAGES, book, respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert (record['value'], record['layer']) == ('urban', 'regex') and respond.sent == []

    def test_no_respond_gives_not_found(self):
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']})[0]
        assert (record['status'], record['layer']) == ('not_found', 'none')

    def test_number_out_of_range_needs_review(self):
        book = load_codebook({'n': {'type': 'integer', 'range': [1, 100], 'query': 'How many schools?'}})
        respond = responder({'value': 500, 'quote': 'two urban school settings', 'passage': 1})
        record = extract_values(SETTING_PAGES, book, respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert record['value'] is None and record['status'] == 'review'

    def test_multi_category_with_unlisted_member_needs_review(self):
        book = load_codebook({'kinds': {'type': 'multi_category', 'values': ['urban', 'rural']}})
        respond = responder({'value': ['urban', 'coastal'], 'quote': 'two urban school settings', 'passage': 1})
        record = extract_values(SETTING_PAGES, book, respond=respond, encoder=Encoder(), **LOCAL)[0]
        assert record['value'] is None and record['status'] == 'review'

    def test_reply_that_is_valid_json_but_not_an_object_needs_review(self):
        record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                                respond=responder('[1, 2]'), encoder=Encoder(), **LOCAL)[0]
        assert record['value'] is None and record['status'] == 'review'


class TestPromptLayerFixes:
    def run(self, book, reply, pages=SETTING_PAGES):
        respond = responder(reply)
        return extract_values(pages, book, respond=respond, encoder=Encoder(), **LOCAL), respond

    def test_brace_values_do_not_break_the_template(self):
        book = load_codebook({'a': {'type': 'category', 'values': ['{x}', 'y']}})
        records, respond = self.run(book, {'value': '{x}', 'quote': 'two urban school settings',
                                           'passage': 1})
        assert records[0]['value'] == '{x}' and '["{x}", "y"]' in respond.sent[0]

    def test_brace_query_is_safe(self):
        book = load_codebook({'a': {'type': 'text', 'query': 'What is {x} in {0}?'}})
        records, respond = self.run(book, {'value': 'urban', 'quote': 'two urban school settings',
                                           'passage': 1})
        assert records[0]['status'] == 'extracted' and 'What is {x} in {0}?' in respond.sent[0]

    def test_integer_answer_is_an_int(self):
        book = load_codebook({'n': {'type': 'integer', 'range': [1, 100], 'query': 'How many?'}})
        records, _ = self.run(book, {'value': 48.0, 'quote': 'two urban school settings',
                                     'passage': 1})
        assert records[0]['value'] == 48 and isinstance(records[0]['value'], int)

    def test_multi_category_gives_one_record_per_value(self):
        book = load_codebook({'kinds': {'type': 'multi_category', 'values': ['urban', 'rural']}})
        records, _ = self.run(book, {'value': ['urban', 'rural'],
                                     'quote': 'two urban school settings', 'passage': 1})
        assert [r['value'] for r in records] == ['urban', 'rural']
        assert {r['quote'] for r in records} == {'two urban school settings'}
        assert {r['pdf_page'] for r in records} == {3}

    def test_null_answer_is_not_found(self):
        records, _ = self.run({'region': TEXT_BOOK['region']},
                              {'value': None, 'quote': '', 'passage': 1})
        assert (records[0]['status'], records[0]['layer'], records[0]['value']) == (
            'not_found', 'prompt', None)


def test_record_has_paragraph_and_table_keys():
    record = extract_values(PAGES, BOOK)[0]
    assert record['paragraph'] is None and record['table'] is None
    assert list(record)[list(record).index('printed_page') + 1:][:2] == ['paragraph', 'table']


def test_docx_blocks_keep_paragraph_and_no_pdf_page(tmp_path):
    docx = pytest.importorskip('docx')
    from pubextractor.docx_reader import read_docx
    document = docx.Document()
    document.add_heading('Method', level=1)
    document.add_paragraph('Participants were young learners.')
    document.add_paragraph('The study took place in two urban school settings.')
    path = tmp_path / 'study.docx'
    document.save(path)
    pages = read_docx(path)
    book = load_codebook({
        'grade_band': {'type': 'category', 'values': ['middle', 'high'],
                       'terms': {'middle': ['young learners']}},
        'setting': {'type': 'text', 'query': 'Where did the study take place?'}})
    respond = responder({'value': 'two urban schools',
                         'quote': 'The study took place in two urban school settings.',
                         'passage': 3})
    records = extract_values(pages, book, respond=respond, encoder=Encoder(), **LOCAL)
    by = {r['field']: r for r in records}
    assert (by['grade_band']['paragraph'], by['grade_band']['pdf_page']) == (2, None)
    assert by['setting']['status'] == 'extracted'
    assert (by['setting']['paragraph'], by['setting']['pdf_page']) == (3, None)
    assert by['setting']['section'] == 'Method'


def test_choice_matches_the_list_ignoring_case_and_spaces():
    respond = responder({'value': ' Urban ', 'quote': 'The study took place in two urban school settings.',
                         'passage': 1})
    record = extract_values(SETTING_PAGES, {'region': TEXT_BOOK['region']},
                            respond=respond, encoder=Encoder(), **LOCAL)[0]
    assert (record['value'], record['status']) == ('urban', 'extracted')


class TestHeadingLines:
    TEXT = ('Method\nThe sample was N = 46 students in middle school.\n'
            'Table 1 lists the measures\nResults\nScores rose, t(44) = 2.10.')

    def test_heading_is_not_glued_to_the_quote(self):
        pages = detect_sections([{'page': 1, 'text': self.TEXT}])
        record = by_field(extract_values(pages, BOOK))['sample_size'][0]
        assert record['quote'] == 'The sample was N = 46 students in middle school.'
        assert record['section'] == 'Method'

    def test_offsets_point_at_the_sentence_and_its_section(self):
        from pubextractor.fulltext import section_at
        from pubextractor.layers import _sentences
        pages = detect_sections([{'page': 1, 'text': self.TEXT}])
        sentences = _sentences(pages)
        flat = self.TEXT.replace('\n', ' ')
        assert all(flat[s['start']:s['start'] + len(s['text'])] == s['text'] for s in sentences)
        assert [s['text'] for s in sentences][-1] == 'Scores rose, t(44) = 2.10.'
        assert not any(s['text'].startswith(('Method', 'Results')) for s in sentences)
        assert [section_at(s['page'], s['start']) for s in sentences] == ['Method', 'Method',
                                                                         'Results']

    def test_heading_line_splits_without_section_marks(self):
        pages = [{'page': 1, 'text': 'Participants\nMiddle school students took part.'}]
        record = by_field(extract_values(pages, BOOK))['grade_band'][0]
        assert record['quote'] == 'Middle school students took part.'
