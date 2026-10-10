"""Read DOCX paragraphs, tables, and images with their position in the document."""


def _open(path):
    try:
        import docx
    except ImportError:
        raise ImportError('DOCX reading requires python-docx; reinstall pubextractor') from None
    from docx.opc.exceptions import PackageNotFoundError
    try:
        return docx.Document(str(path))
    except PackageNotFoundError:
        raise ValueError('only .docx is supported; convert .doc to .docx') from None


W_VAL = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val'


def _tag(element):
    return element.tag.rsplit('}', 1)[-1]


def _body_blocks(parent):
    for element in parent.iterchildren():
        if _tag(element) == 'sdt':
            content = element.find('{*}sdtContent')
            yield from (_body_blocks(content) if content is not None else ())
        else:
            yield element


def _row_texts(row, table):
    from docx.table import _Cell
    texts = []
    for tc in row._tr.tc_lst:
        props = tc.tcPr
        merge = props.find('{*}vMerge') if props is not None else None
        span = props.find('{*}gridSpan') if props is not None else None
        continued = merge is not None and merge.get(W_VAL) != 'restart'
        texts += ['' if continued else _Cell(tc, table).text]
        texts += [''] * (int(span.get(W_VAL)) - 1 if span is not None else 0)
    return texts


def read_docx(path):
    """Blocks of a DOCX file in body order, each with its section and number.

    A DOCX has no fixed pages, so a block is located by section, paragraph
    number, and table number instead.
    """
    document = _open(path)
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    blocks, section, counts = [], '', {'paragraph': 0, 'table': 0}
    for element in _body_blocks(document.element.body):
        tag = _tag(element)
        if tag == 'p':
            paragraph = Paragraph(element, document)
            style = (paragraph.style.name or '') if paragraph.style is not None else ''
            if style.startswith('Heading') and paragraph.text.strip():
                section = paragraph.text.strip()
            counts['paragraph'] += 1
            blocks.append({'kind': 'paragraph', 'paragraph': counts['paragraph'],
                           'section': section, 'text': paragraph.text})
        elif tag == 'tbl':
            table = Table(element, document)
            counts['table'] += 1
            blocks.append({'kind': 'table', 'table': counts['table'], 'section': section,
                           'rows': [_row_texts(row, table) for row in table.rows]})
    return blocks


BLIP = '{http://schemas.openxmlformats.org/drawingml/2006/main}blip'
EMBED = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'


def read_docx_images(path):
    """Images embedded in a DOCX file, numbered in the order they appear."""
    document = _open(path)
    parts = document.part.related_parts
    ids = [blip.get(EMBED) for blip in document.element.body.iter(BLIP)]
    images = [parts[rid] for rid in ids if rid in parts]
    return [{'figure': number, 'content_type': part.content_type, 'content': part.blob}
            for number, part in enumerate(images, start=1)]
