import pytest

docx = pytest.importorskip('docx')

from pubextractor.docx_reader import read_docx, read_docx_images  # noqa: E402


def build(tmp_path):
    document = docx.Document()
    document.add_heading('Method', level=1)
    document.add_paragraph('Fifty students took part.')
    table = document.add_table(rows=2, cols=3)
    for col, text in enumerate(['Group', 'M', 'SD']):
        table.cell(0, col).text = text
    for col, text in enumerate(['Treatment', '61.3', '14.2']):
        table.cell(1, col).text = text
    document.add_heading('Results', level=1)
    document.add_paragraph('Scores rose.')
    path = tmp_path / 'study.docx'
    document.save(path)
    return path


class TestReadDocx:
    def test_blocks_keep_document_order_and_sections(self, tmp_path):
        blocks = read_docx(build(tmp_path))
        kinds = [(b['kind'], b['section']) for b in blocks if b.get('text') != '']
        assert kinds == [('paragraph', 'Method'), ('paragraph', 'Method'), ('table', 'Method'),
                         ('paragraph', 'Results'), ('paragraph', 'Results')]

    def test_table_cells_are_read(self, tmp_path):
        table = [b for b in read_docx(build(tmp_path)) if b['kind'] == 'table'][0]
        assert table['table'] == 1
        assert table['rows'] == [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2']]

    def test_paragraphs_are_numbered(self, tmp_path):
        paragraphs = [b for b in read_docx(build(tmp_path)) if b['kind'] == 'paragraph']
        assert [p['paragraph'] for p in paragraphs] == list(range(1, len(paragraphs) + 1))
        assert paragraphs[1]['text'] == 'Fifty students took part.'


def tiny_png(tmp_path, name, color):
    pytest.importorskip('PIL')
    from PIL import Image

    path = tmp_path / name
    Image.new('RGB', (4, 4), color).save(path)
    return path


class TestReadDocxImages:
    def test_images_in_body_order(self, tmp_path):
        document = docx.Document()
        document.add_paragraph('Figure 1')
        document.add_picture(str(tiny_png(tmp_path, 'red.png', 'red')))
        document.add_paragraph('Figure 2')
        document.add_picture(str(tiny_png(tmp_path, 'blue.png', 'blue')))
        path = tmp_path / 'figures.docx'
        document.save(path)
        images = read_docx_images(path)
        assert [i['figure'] for i in images] == [1, 2]
        assert all(i['content_type'] == 'image/png' for i in images)
        assert images[0]['content'] == (tmp_path / 'red.png').read_bytes()

    def test_document_without_images(self, tmp_path):
        document = docx.Document()
        document.add_paragraph('No figures.')
        path = tmp_path / 'plain.docx'
        document.save(path)
        assert read_docx_images(path) == []
