import io
import json

import pytest

pytest.importorskip('reportlab')
pytest.importorskip('pdfplumber')
Image = pytest.importorskip('PIL.Image')

from pubextractor import (  # noqa: E402
    STAT_ROLES,
    ProvenanceTracker,
    build_comparisons,
    check_llm_use,
    effect_sizes,
    extract_table_values,
    find_image_tables,
    manual_table_records,
    prisma_items,
    read_table_image,
    read_tables,
    table_image,
)

PAGE_HEIGHT = 792
PROSE = 'Scores rose over the term in both groups of pupils'


def grid_picture(width=400, height=150):
    from PIL import ImageDraw
    picture = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(picture)
    [draw.line([(0, y), (width, y)], fill='black') for y in range(0, height, 30)]
    [draw.line([(x, 0), (x, height)], fill='black') for x in range(0, width, 100)]
    [draw.text((x + 10, y + 10), f'{x // 10 + y // 3}.5', fill='black')
     for x in range(100, width, 100) for y in range(30, height, 30)]
    buffer = io.BytesIO()
    picture.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def image_pdf(tmp_path):
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    def page(sheet, layout):
        x, width = layout['image']
        sheet.drawString(layout['caption_x'], 700, layout['caption'])
        [sheet.drawString(layout['caption_x'], 700 - 14 * (i + 1), line)
         for i, line in enumerate(layout.get('more', []))]
        sheet.drawImage(ImageReader(grid_picture(int(width), 150)), x, 520, width, 150)
        sheet.drawString(x, 505, 'Note. Higher scores are better.')
        [sheet.drawString(column, y, PROSE[:40]) for column in layout.get('beside', [])
         for y in range(500, 760, 14)]
        [sheet.drawString(column, y, PROSE[:40]) for column in layout.get('below', [72])
         for y in range(200, 470, 14)]
        sheet.showPage()

    def build(*layouts):
        path = tmp_path / 'scanned.pdf'
        sheet = canvas.Canvas(str(path), pagesize=(612, PAGE_HEIGHT))
        [page(sheet, layout) for layout in layouts]
        sheet.save()
        return path
    return build


SINGLE = {'caption': 'Table 1. Posttest scores by group', 'caption_x': 72, 'image': (72, 400),
          'more': ['Means and standard deviations']}
LEFT = {'caption': 'Table 2. Scores', 'caption_x': 72, 'image': (72, 210),
        'beside': [330], 'below': [72, 330]}
WIDE = {'caption': 'Table 3. Scores', 'caption_x': 72, 'image': (72, 460), 'below': [72, 330]}


class TestFindImageTables:
    def test_caption_over_picture_is_reported(self, image_pdf):
        found = find_image_tables(image_pdf(SINGLE))
        assert len(found) == 1
        table = found[0]
        assert (table['table'], table['pdf_page'], table['printed_page']) == (1, 1, None)
        assert table['caption'] == 'Table 1. Posttest scores by group Means and standard deviations'
        assert table['note'] == 'Note. Higher scores are better.'
        assert table['reason'] == 'table body is an image'

    def test_region_runs_from_caption_to_note(self, image_pdf):
        x0, top, x1, bottom = find_image_tables(image_pdf(SINGLE))[0]['bbox']
        assert top <= PAGE_HEIGHT - 700 - 6
        assert PAGE_HEIGHT - 520 <= bottom <= PAGE_HEIGHT - 505 - 6
        assert x0 <= 72 and x1 >= 472

    def test_region_keeps_to_caption_column(self, image_pdf):
        x0, _, x1, _ = find_image_tables(image_pdf(LEFT))[0]['bbox']
        assert x0 <= 72 and 282 <= x1 < 330

    def test_full_width_picture_on_two_column_page_spans_page(self, image_pdf):
        x0, _, x1, _ = find_image_tables(image_pdf(WIDE))[0]['bbox']
        assert x0 <= 72 and x1 >= 532

    def test_numbers_run_across_pages(self, image_pdf):
        found = find_image_tables(image_pdf(SINGLE, WIDE))
        assert [(t['table'], t['pdf_page']) for t in found] == [(1, 1), (2, 2)]

    @pytest.mark.parametrize('caption', ['TABLE1. Scores', 'Table1 Scores'])
    def test_caption_without_space_is_reported(self, image_pdf, caption):
        found = find_image_tables(image_pdf({**SINGLE, 'caption': caption}))
        assert [t['caption'] for t in found] == [f'{caption} Means and standard deviations']

    def test_text_table_is_not_an_image_table(self, borderless_pdf):
        rows = [['Group', 'M', 'SD'], ['Treatment', '61.3', '14.2'], ['Control', '55.0', '13.1']]
        assert find_image_tables(borderless_pdf(['Table 1', rows, PROSE * 2])) == []

    def test_caption_over_blank_band_without_picture_is_not_reported(self, tmp_path):
        from reportlab.pdfgen import canvas
        path = tmp_path / 'blank.pdf'
        sheet = canvas.Canvas(str(path), pagesize=(612, PAGE_HEIGHT))
        sheet.drawString(72, 700, 'Table 1. Scores')
        sheet.drawString(72, 400, 'Note. Nothing between.')
        sheet.save()
        assert find_image_tables(path) == []


class TestReadTablesFlagsImages:
    def test_grid_has_no_rows_and_strategy_image(self, image_pdf):
        grids = read_tables(image_pdf(SINGLE))
        assert [(g['table'], g['strategy'], g['rows']) for g in grids] == [(1, 'image', [])]
        assert grids[0]['reason'] == 'table body is an image'

    def test_extraction_gives_one_flagged_table_record(self, image_pdf):
        records = extract_table_values(read_tables(image_pdf(SINGLE)), paper_id='S01')
        assert len(records) == 1
        record = records[0]
        assert (record['field'], record['status'], record['value']) == ('table', 'not_found', None)
        assert record['quote'] == ('table 1 on page 1 is an image: enter values or read with '
                                   'a vision model')


class TestTableImage:
    def test_crop_is_png_at_resolution(self, image_pdf):
        path = image_pdf(SINGLE)
        bbox = find_image_tables(path)[0]['bbox']
        png = table_image(path, 1, bbox, resolution=144)
        picture = Image.open(io.BytesIO(png))
        assert png[:8] == b'\x89PNG\r\n\x1a\n'
        assert abs(picture.width - (bbox[2] - bbox[0]) * 2) <= 2
        assert abs(picture.height - (bbox[3] - bbox[1]) * 2) <= 2

    @pytest.mark.parametrize('suffix', ['.png', '.jpg'])
    def test_image_file_is_one_table_image(self, tmp_path, suffix):
        path = tmp_path / f'table-2{suffix}'
        Image.open(grid_picture()).save(path)
        grid = read_tables(path)[0]
        assert (grid['table'], grid['strategy'], grid['bbox'], grid['caption']) == (
            1, 'image', (0, 0, 400, 150), 'table-2')
        assert Image.open(io.BytesIO(table_image(path, 1, grid['bbox']))).size == (400, 150)
        assert Image.open(io.BytesIO(table_image(path, 1, (0, 0, 100, 30)))).size == (100, 30)

    def test_transparent_rotated_and_16_bit_images_read_upright_on_white(self, tmp_path):
        import numpy as np
        clear = Image.new('RGBA', (40, 20), (0, 0, 0, 0))
        clear.save(tmp_path / 'clear.png')
        assert Image.open(io.BytesIO(table_image(tmp_path / 'clear.png', 1, (0, 0, 40, 20)))).getpixel((5, 5)) == (255, 255, 255)
        rotated = Image.new('RGB', (40, 20), 'white')
        exif = rotated.getexif()
        exif[0x0112] = 6
        rotated.save(tmp_path / 'photo.jpg', exif=exif)
        assert read_tables(tmp_path / 'photo.jpg')[0]['bbox'] == (0, 0, 20, 40)
        Image.fromarray(np.full((10, 10), 40000, dtype=np.uint16)).save(tmp_path / 'deep.png')
        assert Image.open(io.BytesIO(table_image(tmp_path / 'deep.png', 1, (0, 0, 10, 10)))).getpixel((1, 1)) == (255, 255, 255)

    def test_image_file_records_ask_for_entry(self, tmp_path):
        path = tmp_path / 'scan.png'
        Image.open(grid_picture()).save(path)
        records = extract_table_values(read_tables(path), paper_id='scan')
        assert [(r['status'], r['quote']) for r in records] == [
            ('not_found', 'table 1 on page 1 is an image: enter values or read with a vision model')]

    def test_box_past_the_page_is_clipped(self, image_pdf):
        png = table_image(image_pdf(SINGLE), 1, (-20, 50, 700, 300), resolution=72)
        assert Image.open(io.BytesIO(png)).width <= 612


ENTRY = [
    {'arm': 'Intervention', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'mean',
     'value': '20'},
    {'arm': 'Intervention', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'sd',
     'value': 4},
    {'arm': 'Intervention', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'n',
     'value': '10'},
    {'arm': 'Control', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'mean',
     'value': '16'},
    {'arm': 'Control', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'sd',
     'value': '4'},
    {'arm': 'Control', 'group': '', 'outcome': 'Reading', 'time': 'post', 'stat': 'n',
     'value': '10'},
]
BLANK = {'arm': '', 'group': '', 'outcome': '', 'time': '', 'stat': '', 'value': ''}


class TestManualTableRecords:
    def test_records_follow_table_schema(self):
        records = manual_table_records(ENTRY, 'S01', 3, pdf_page=9, printed_page=89)
        first = records[0]
        assert first['field'] == 'treatment_post_mean'
        assert (first['layer'], first['status'], first['quote']) == ('human', 'extracted',
                                                                     'entered by coder')
        assert (first['table'], first['pdf_page'], first['printed_page']) == (3, 9, 89)
        assert (first['row'], first['col'], first['value']) == (None, None, 20)
        assert set(first) == set(extract_table_values([])[0])

    def test_groups_are_inferred_when_blank(self):
        records = manual_table_records(ENTRY, 'S01', 3)
        assert {r['arm']: r['group'] for r in records} == {'Intervention': 'treatment',
                                                           'Control': 'control'}

    def test_stated_group_wins(self):
        rows = [{**ENTRY[0], 'arm': 'CP', 'group': 'treatment'}]
        assert manual_table_records(rows, 'S01', 1)[0]['group'] == 'treatment'

    def test_blank_rows_are_skipped(self):
        assert len(manual_table_records([BLANK] + ENTRY + [BLANK], 'S01', 3)) == 6

    def test_n_is_an_integer(self):
        records = manual_table_records(ENTRY, 'S01', 3)
        assert [r['value'] for r in records if r['stat'] == 'n'] == [10, 10]
        assert all(isinstance(r['value'], int) for r in records if r['stat'] == 'n')

    def test_mean_sd_cell_gives_two_records(self):
        rows = [{**ENTRY[0], 'stat': 'mean_sd', 'value': '20 (4)'}]
        records = manual_table_records(rows, 'S01', 1)
        assert [(r['stat'], r['value']) for r in records] == [('mean', 20), ('sd', 4)]

    def test_unknown_stat_names_the_row(self):
        with pytest.raises(ValueError, match='row 2'):
            manual_table_records([ENTRY[0], {**ENTRY[1], 'stat': 'median'}], 'S01', 1)

    def test_unreadable_value_names_the_row(self):
        with pytest.raises(ValueError, match='row 1'):
            manual_table_records([{**ENTRY[0], 'value': 'about twenty'}], 'S01', 1)

    def test_unknown_time_names_the_row(self):
        with pytest.raises(ValueError, match='row 1'):
            manual_table_records([{**ENTRY[0], 'time': 'midway'}], 'S01', 1)

    def test_posttest_g_matches_hand_calculation(self):
        effects = effect_sizes(build_comparisons(manual_table_records(ENTRY, 'S01', 3)))
        computed = [e for e in effects if e['g'] is not None]
        assert len(computed) == 1
        # d = (20 - 16) / 4 = 1; J = 1 - 3 / (4 * 18 - 1)
        assert computed[0]['g'] == pytest.approx(1 - 3 / 71)
        assert (computed[0]['n_t'], computed[0]['n_c']) == (10, 10)


PNG = b'\x89PNG\r\n\x1a\nstub'
REPLY = json.dumps({'rows': [
    {'arm': 'Intervention', 'outcome': 'Reading', 'time': 'post', 'stat': 'mean_sd',
     'value': '20 (4)', 'cell_text': '20 (4)'},
    {'arm': 'Control', 'outcome': 'Reading', 'time': 'post', 'stat': 'mean', 'value': 16,
     'cell_text': '16.0'},
    {'arm': 'Control', 'outcome': 'Reading', 'time': 'post', 'stat': 'median', 'value': 15,
     'cell_text': '15'},
    {'arm': 'Control', 'outcome': 'Reading', 'time': 'post', 'stat': 'sd', 'value': 5,
     'cell_text': '4'},
]})


def responder(reply=REPLY, failure=None):
    calls = []

    def respond(prompt, images=None):
        calls.append({'prompt': prompt, 'images': images})
        if failure:
            raise failure
        return reply
    respond.model = 'stub-vision'
    return respond, calls


LOCAL = check_llm_use(access='subscription', ai_terms='silent', processing='local')


class TestReadTableImage:
    def test_refusal_sends_nothing(self):
        respond, calls = responder()
        permission = check_llm_use(access='subscription', ai_terms='forbids', processing='api')
        records = read_table_image(PNG, respond, permission, 'https://api.example.org/v1',
                                   paper_id='S01', table=3, pdf_page=9)
        assert calls == []
        assert len(records) == 1
        assert records[0]['status'] == 'refused'
        assert records[0]['quote']
        assert (records[0]['table'], records[0]['pdf_page']) == (3, 9)

    def test_local_route_needs_local_endpoint(self):
        respond, calls = responder()
        records = read_table_image(PNG, respond, LOCAL, 'https://api.example.org/v1')
        assert calls == [] and records[0]['status'] == 'refused'

    def test_edited_permission_is_refused(self):
        respond, calls = responder()
        edited = {**check_llm_use(ai_terms='forbids'), 'allowed': True, 'verdict': 'any_llm'}
        records = read_table_image(PNG, respond, edited, 'http://localhost:11434/v1')
        assert calls == [] and records[0]['status'] == 'refused'

    def test_one_prompt_carries_the_image_and_closed_list(self):
        respond, calls = responder()
        read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1')
        assert len(calls) == 1 and calls[0]['images'] == [PNG]
        assert json.dumps(list(STAT_ROLES)) in calls[0]['prompt']
        assert 'cell_text' in calls[0]['prompt'] and 'JSON' in calls[0]['prompt']

    def test_kept_values_are_review_with_cell_quote(self):
        respond, _ = responder()
        records = read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1',
                                   paper_id='S01', table=3, pdf_page=9)
        kept = [r for r in records if r['stat']]
        assert [(r['stat'], r['value'], r['quote']) for r in kept] == [
            ('mean', 20, '20 (4)'), ('sd', 4, '20 (4)'), ('mean', 16, '16.0')]
        assert {(r['layer'], r['status']) for r in records} == {('vision', 'review')}
        assert {r['field'] for r in kept} == {'treatment_post_mean', 'treatment_post_sd',
                                             'control_post_mean'}
        assert {(r['table'], r['pdf_page']) for r in records} == {(3, 9)}

    def test_unknown_stat_and_mismatch_are_dropped_and_flagged(self):
        respond, _ = responder()
        records = read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1')
        dropped = [r for r in records if not r['stat']]
        assert len(dropped) == 2
        assert all(r['value'] is None and r['status'] == 'review' for r in dropped)
        assert 'median' in dropped[0]['quote'] and '4' in dropped[1]['quote']

    def test_failed_call_gives_review(self):
        respond, _ = responder(failure=RuntimeError('down'))
        records = read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1')
        assert [(r['status'], r['quote']) for r in records] == [
            ('review', 'model call failed: RuntimeError')]

    def test_unreadable_reply_gives_review(self):
        respond, _ = responder(reply='I cannot read this table.')
        records = read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1')
        assert [(r['status'], r['value']) for r in records] == [('review', None)]

    def test_tracker_keeps_prompt_hash_and_model_not_image(self):
        respond, _ = responder()
        tracker = ProvenanceTracker('extraction', 'tables')
        read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1', paper_id='S01',
                         tracker=tracker)
        log = tracker.to_dict()
        assert log['prompts']['records'][0]['model'] == 'stub-vision'
        assert log['permissions']['S01']['verdict'] == LOCAL['verdict']
        assert 'stub' not in json.dumps(log, default=str).replace('stub-vision', '')

    def test_confirmed_vision_values_name_the_model_in_methods(self):
        respond, _ = responder()
        tracker = ProvenanceTracker('extraction', 'tables')
        read_table_image(PNG, respond, LOCAL, 'http://localhost:11434/v1', paper_id='S01',
                         tracker=tracker)
        records = manual_table_records(ENTRY, 'S01', 3, layer='vision')
        assert {(r['layer'], r['status']) for r in records} == {('vision', 'extracted')}
        assert 'confirmed by coder' in records[0]['quote']
        items = prisma_items(tracker, records)
        assert 'stub-vision' in items['item 9'] and 'stub-vision' in items['item 11']
        assert 'by a vision model' in items['item 9']
        assert LOCAL['verdict'] in items['item 9']
