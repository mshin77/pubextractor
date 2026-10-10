import pytest


@pytest.fixture
def study_docx(tmp_path):
    docx = pytest.importorskip('docx')
    document = docx.Document()
    document.add_heading('Results', level=1)
    document.add_paragraph('Scores rose in both groups.')
    document.add_paragraph('Table 1')
    document.add_paragraph('Reading scores by group')
    table = document.add_table(rows=3, cols=5)
    table.cell(0, 1).merge(table.cell(0, 2))
    table.cell(0, 3).merge(table.cell(0, 4))
    table.cell(0, 1).text = 'Treatment (n = 24)'
    table.cell(0, 3).text = 'Control (n = 22)'
    for row, texts in ((1, ['Measure', 'M', 'SD', 'M', 'SD']),
                       (2, ['Reading', '61.3', '14.2', '55.0', '13.1'])):
        for col, text in enumerate(texts):
            table.cell(row, col).text = text
    document.add_paragraph('Note. Higher scores mean better reading.')
    document.add_paragraph('Discussion follows.')
    path = tmp_path / 'study.docx'
    document.save(path)
    return path


@pytest.fixture
def study_pdf(tmp_path):
    pytest.importorskip('reportlab')
    pytest.importorskip('pdfplumber')
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

    def build(rows, caption=None, note=None):
        style = getSampleStyleSheet()['Normal']
        table = Table(rows)
        table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.8, (0, 0, 0))]))
        story = ([Paragraph(text, style) for text in (caption or [])] + [table]
                 + [Paragraph(text, style) for text in (note or [])])
        path = tmp_path / 'study.pdf'
        SimpleDocTemplate(str(path), pagesize=letter).build(story)
        return path
    return build


@pytest.fixture
def two_column_pdf(tmp_path):
    pytest.importorskip('reportlab')
    pytest.importorskip('pdfplumber')
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph

    def build(title, left, right):
        styles = getSampleStyleSheet()
        frames = [Frame(72, 650, 468, 70, id='top'), Frame(72, 72, 222, 570, id='left'),
                  Frame(318, 72, 222, 570, id='right')]
        story = ([Paragraph(title, styles['Normal']), FrameBreak()]
                 + [Paragraph(text, styles['Heading2'] if len(text.split()) == 1
                              else styles['Normal']) for text in left]
                 + [FrameBreak()]
                 + [Paragraph(text, styles['Heading2'] if len(text.split()) == 1
                              else styles['Normal']) for text in right])
        path = tmp_path / 'columns.pdf'
        document = BaseDocTemplate(str(path), pagesize=letter)
        document.addPageTemplates([PageTemplate(id='page', frames=frames)])
        document.build(story)
        return path
    return build


@pytest.fixture
def borderless_pdf(tmp_path):
    pytest.importorskip('reportlab')
    pytest.importorskip('pdfplumber')
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        FrameBreak,
        PageTemplate,
        Paragraph,
        Table,
    )

    def flow(items, styles, table_style):
        return [Table(item, hAlign='LEFT', style=table_style) if isinstance(item, list)
                else FrameBreak() if item is None
                else Paragraph(item, styles['Heading2'] if len(item.split()) == 1
                               else styles['Normal'])
                for item in items]

    def build(items, columns=1, table_style=None):
        styles = getSampleStyleSheet()
        frames = ([Frame(72, 72, 468, 648, id='body')] if columns == 1 else
                  [Frame(72, 72, 222, 648, id='left'), Frame(318, 72, 222, 648, id='right')])
        path = tmp_path / 'borderless.pdf'
        document = BaseDocTemplate(str(path), pagesize=letter)
        document.addPageTemplates([PageTemplate(id='page', frames=frames)])
        document.build(flow(items, styles, table_style))
        return path
    return build
