import pytest

from pubextractor.fulltext import SECTION_NAMES, detect_page_labels, detect_sections, format_anchor


def page(number, text):
    return {'page': number, 'text': text}


class TestPageLabels:
    def test_reads_page_of_running_head(self):
        pages = [page(1, 'Author et al. Journal (2026) 15:185\nbody'),
                 page(2, 'Author et al. Journal (2026) 15:185 Page 2 of 14\nbody'),
                 page(3, 'Author et al. Journal (2026) 15:185 Page 3 of 14\nbody')]
        assert [p['printed_page'] for p in detect_page_labels(pages)] == [1, 2, 3]

    def test_reads_offset_numbering(self):
        pages = [page(1, 'body\n576'), page(2, 'body\n577'), page(3, 'body\n578')]
        labeled = detect_page_labels(pages)
        assert [p['printed_page'] for p in labeled] == [576, 577, 578]
        assert [p['page'] for p in labeled] == [1, 2, 3]

    def test_back_fills_unlabeled_first_page(self):
        pages = [page(1, 'title page, no number'),
                 page(2, 'body\n188'),
                 page(3, 'body\n189')]
        assert [p['printed_page'] for p in detect_page_labels(pages)] == [187, 188, 189]

    def test_inconsistent_numbers_fall_back(self):
        pages = [page(1, 'body\n5'), page(2, 'body\n99'), page(3, 'body\n7'), page(4, 'body\n40')]
        assert all(p['printed_page'] is None for p in detect_page_labels(pages))

    def test_no_numbers_fall_back(self):
        pages = [page(1, 'body only'), page(2, 'body only')]
        assert all(p['printed_page'] is None for p in detect_page_labels(pages))


class TestSections:
    def test_heading_lines_set_section_per_page(self):
        pages = [page(1, 'Abstract\nBackground\nSystematic reviews are essential'),
                 page(2, 'Results\nThe model recovered every include')]
        labeled = detect_sections(pages)
        assert labeled[0]['section'] == 'Background'
        assert labeled[1]['section'] == 'Results'

    def test_unmatched_heading_keeps_previous_label(self):
        pages = [page(1, 'Methods\nwe did things'), page(2, 'Some Idiosyncratic Heading')]
        assert [p['section'] for p in detect_sections(pages)] == ['Methods', 'Methods']

    def test_trailing_symbol_noise_does_not_hide_heading(self):
        pages = [page(1, 'Results �\nScores rose.'), page(2, 'Discussion ~;\nGains held.')]
        assert [p['section'] for p in detect_sections(pages)] == ['Results', 'Discussion']

    def test_vocabulary_is_not_empty(self):
        assert 'discussion' in SECTION_NAMES


class TestAnchor:
    def test_prefers_printed_page(self):
        span = {'page': 188, 'pdf_page': 2, 'printed_page': 188, 'section': 'Results'}
        assert format_anchor(span) == 'p. 188 · Results'

    def test_marks_file_position_when_printed_unknown(self):
        span = {'page': 2, 'pdf_page': 2, 'printed_page': None, 'section': 'Results'}
        assert format_anchor(span) == 'file p. 2 · Results'

    def test_omits_absent_section(self):
        span = {'page': 5, 'pdf_page': 5, 'printed_page': 5, 'section': ''}
        assert format_anchor(span) == 'p. 5'


class TestSimilarityScores:
    class Encoder:
        def encode(self, texts):
            import numpy as np
            return np.array([[float('fraction' in t), float('reading' in t)] for t in texts])

    def test_matching_text_scores_highest(self):
        from pubextractor.fulltext import similarity_scores
        scores = similarity_scores(['fraction tasks', 'reading tasks'], 'fraction', self.Encoder())
        assert scores[0] > scores[1]

    def test_failed_encoding_returns_empty(self):
        from pubextractor.fulltext import similarity_scores

        class Broken:
            def encode(self, texts):
                raise RuntimeError('no model')

        assert similarity_scores(['text'], 'description', Broken()) == []


TITLE = 'Tutoring and reading growth in two middle schools across one full school year of study'
WRAPPED = ('The sample comprised forty pupils drawn from two schools that agreed to take part '
           'in the trial during the autumn term.')
LEFT = ['Method', WRAPPED + ' Pupils were matched on prior attainment before allocation. '
        'Teachers delivered the lessons from a written script in each session. '
        'Attendance was recorded at every lesson by a second observer.']
RIGHT = ['Results', 'Scores rose in the tutored group after eight weeks of lessons. '
         'The comparison group showed little change across the same period. '
         'Gains were similar for boys and girls in both schools during the trial.']


def flat(text):
    return ' '.join(text.split())


class TestColumns:
    def test_two_columns_read_left_then_right(self, two_column_pdf):
        from pubextractor.fulltext import read_pdf
        page = read_pdf(two_column_pdf(TITLE, LEFT, RIGHT))[0]
        text = flat(page['text'])
        assert page['columns'] == 2
        assert WRAPPED in text
        assert (text.index(TITLE) < text.index('Method') < text.index('second observer')
                < text.index('Results') < text.index('both schools'))

    def test_column_headings_are_detected(self, two_column_pdf):
        from pubextractor.fulltext import read_pdf
        page = detect_sections(read_pdf(two_column_pdf(TITLE, LEFT, RIGHT)))[0]
        assert [name for _, name in page['section_marks']] == ['Method', 'Results']
        assert all(page['text'][offset:].startswith(name)
                   for offset, name in page['section_marks'])

    def test_single_column_page_is_unchanged(self, study_pdf):
        pdfplumber = pytest.importorskip('pdfplumber')
        from pubextractor.fulltext import read_pdf
        path = study_pdf([['Group', 'M'], ['Treatment', '5.0']],
                         caption=['Table 1', WRAPPED + ' ' + WRAPPED])
        with pdfplumber.open(path) as document:
            expected = document.pages[0].extract_text()
        page = read_pdf(path)[0]
        assert (page['columns'], page['text']) == (1, expected)


OCR_PHRASE = 'buying a small pet and building a home for it'


def ocr_pdf(path, columns):
    pytest.importorskip('reportlab')
    from itertools import accumulate

    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas
    words = OCR_PHRASE.split()
    steps = [stringWidth(word, 'Helvetica', 10) + 2 for word in words]
    sheet = canvas.Canvas(str(path), pagesize=(612, 792))
    sheet.setFont('Helvetica', 10)
    [sheet.drawString(x, y, word) for left in columns for y in range(700, 580, -14)
     for x, word in zip(accumulate([left] + steps), words)]
    sheet.save()
    return path


class TestWordSpacing:
    @pytest.mark.parametrize('columns', [(72,), (72, 330)])
    def test_close_set_ocr_words_stay_apart(self, tmp_path, columns):
        pytest.importorskip('pdfplumber')
        from pubextractor.fulltext import read_pdf
        page = read_pdf(ocr_pdf(tmp_path / 'ocr.pdf', columns))[0]
        assert page['columns'] == len(columns)
        assert set(page['text'].splitlines()) == {OCR_PHRASE}

    def test_tracked_letters_stay_one_word(self, tmp_path):
        pytest.importorskip('reportlab')
        pytest.importorskip('pdfplumber')
        from reportlab.pdfgen import canvas

        from pubextractor.fulltext import read_pdf
        path = tmp_path / 'tracked.pdf'
        sheet = canvas.Canvas(str(path))
        line = sheet.beginText(72, 700)
        line.setCharSpace(1)
        line.textLine('Contextualized problem instruction')
        sheet.drawText(line)
        sheet.save()
        assert read_pdf(path)[0]['text'] == 'Contextualized problem instruction'
