import pytest

from pubextractor.codebook import import_extraction_schema, load_codebook

FIELDS = {
    'grade_band': {'type': 'category', 'values': ['elementary', 'middle', 'high'],
                   'definition': 'Grade band of participants.',
                   'terms': {'middle': ['middle school', 'grade* 6']}},
    'sample_size': {'type': 'integer', 'stat': 'n', 'range': [1, 100000]},
    'setting': {'type': 'text', 'query': 'Where did the study take place?'},
}


class TestLoadCodebook:
    def test_dict_is_normalized(self):
        book = load_codebook(FIELDS)
        assert book['grade_band']['terms'] == {'middle': ['middle school', 'grade* 6']}
        assert book['sample_size']['range'] == (1, 100000)
        assert book['setting']['values'] == []
        assert book['setting']['query'] == 'Where did the study take place?'

    def test_yaml_file(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_text('design:\n  type: category\n  values: [group, single_case]\n',
                        encoding='utf-8')
        assert load_codebook(path)['design']['values'] == ['group', 'single_case']

    @pytest.mark.parametrize('field, message', [
        ({'type': 'color'}, 'type must be'),
        ({'type': 'category'}, 'values are required'),
        ({'type': 'number', 'stat': 'median'}, 'stat must be'),
        ({'type': 'number', 'range': [1]}, 'range must be'),
        ({'type': 'category', 'values': ['a'], 'terms': {'b': ['x']}}, 'terms name values'),
    ])
    def test_problems_are_reported(self, field, message):
        with pytest.raises(ValueError, match=message):
            load_codebook({'f': field})

    def test_every_problem_is_listed(self):
        with pytest.raises(ValueError) as caught:
            load_codebook({'a': {'type': 'x'}, 'b': {'type': 'category'}})
        assert 'a:' in str(caught.value) and 'b:' in str(caught.value)


class TestMalformedInput:
    def test_top_level_list_is_rejected(self):
        with pytest.raises(ValueError, match='Codebook must be a mapping'):
            load_codebook([{'type': 'text'}])

    def test_yaml_string_file(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_text('just text', encoding='utf-8')
        with pytest.raises(ValueError, match='Codebook must be a mapping'):
            load_codebook(str(path))

    def test_yaml_list_file(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_text('- type: text\n', encoding='utf-8')
        with pytest.raises(ValueError, match='Codebook must be a mapping'):
            load_codebook(path)

    def test_empty_yaml_is_refused(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_text('', encoding='utf-8')
        with pytest.raises(ValueError, match='codebook has no fields'):
            load_codebook(path)

    def test_bom_yaml_file(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_bytes(b'\xef\xbb\xbfdesign:\n  type: text\n')
        assert load_codebook(path)['design']['type'] == 'text'

    @pytest.mark.parametrize('field', [None, 'text', ['text']])
    def test_field_must_be_mapping(self, field):
        with pytest.raises(ValueError, match='f: field must be a mapping'):
            load_codebook({'f': field})

    @pytest.mark.parametrize('kind', [['text'], {'a': 1}, 3, None])
    def test_unhashable_or_non_string_type(self, kind):
        with pytest.raises(ValueError, match='type must be one of'):
            load_codebook({'f': {'type': kind}})

    def test_none_values_and_terms_are_empty(self):
        book = load_codebook({'f': {'type': 'text', 'values': None, 'terms': None}})
        assert book['f']['values'] == [] and book['f']['terms'] == {}

    def test_values_must_be_list(self):
        with pytest.raises(ValueError, match='f: values must be a list'):
            load_codebook({'f': {'type': 'category', 'values': 'abc'}})

    @pytest.mark.parametrize('terms', [{'a': 'x'}, {'a': [1]}, ['a']])
    def test_terms_must_map_to_term_lists(self, terms):
        with pytest.raises(ValueError, match='f: terms must map values to lists of terms'):
            load_codebook({'f': {'type': 'category', 'values': ['a'], 'terms': terms}})

    @pytest.mark.parametrize('span', [[5, 1], ['a', 'b'], [True, 3], 'ab', {'low': 1}])
    def test_range_needs_ordered_numbers(self, span):
        with pytest.raises(ValueError, match=r'range must be \[low, high\] with numbers, low <= high'):
            load_codebook({'f': {'type': 'number', 'range': span}})

    def test_stat_none_is_allowed(self):
        assert load_codebook({'f': {'type': 'number', 'stat': None}})['f']['stat'] is None

    def test_offending_value_is_named(self):
        with pytest.raises(ValueError, match="not 'color'"):
            load_codebook({'f': {'type': 'color'}})
        with pytest.raises(ValueError, match="not 'median'"):
            load_codebook({'f': {'type': 'number', 'stat': 'median'}})


class TestYamlBooleans:
    def test_yes_no_values_and_terms_become_text(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'codebook.yml'
        path.write_text('flag:\n  type: category\n  values: [yes, no]\n'
                        '  terms:\n    yes: [randomized]\n', encoding='utf-8')
        field = load_codebook(path)['flag']
        assert field['values'] == ['yes', 'no']
        assert field['terms'] == {'yes': ['randomized']}

    def test_numeric_values_become_text(self):
        field = load_codebook({'f': {'type': 'category', 'values': [1, 2],
                                     'terms': {1: ['one']}}})['f']
        assert field['values'] == ['1', '2']
        assert field['terms'] == {'1': ['one']}

    @pytest.mark.parametrize('bad', [[['a']], [{'a': 1}]])
    def test_non_scalar_values_are_rejected(self, bad):
        with pytest.raises(ValueError, match='values must be text'):
            load_codebook({'f': {'type': 'multi_category', 'values': bad}})


class TestMissingSource:
    def test_yaml_text_string_names_the_hint(self):
        with pytest.raises(FileNotFoundError, match='load_codebook takes a dict or a YAML file path'):
            load_codebook('design:\n  type: text\n')

    def test_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='nope.yml'):
            load_codebook(tmp_path / 'nope.yml')


SCHEMA = {
    'vocabulary': {'type': 'binary', 'query': 'Does the study examine vocabulary?',
                   'description': 'Whether vocabulary is examined',
                   'stage1_alignment': "Matches 'vocabular'"},
    'vocabulary_focus': {'type': 'categorical_multiple',
                         'values': ['technical_terms', 'word_learning', 'not_applicable'],
                         'query': 'What aspect of vocabulary?'},
    'sample_size': {'type': 'number', 'query': 'How many participants?'},
    'grade_level': {'type': 'categorical', 'values': ['elementary', 'middle']},
    'vocabulary_evidence': {'type': 'text', 'query': 'Quote the evidence.'},
    'linguistic_count': {'type': 'derived', 'formula': 'sum'},
}


class TestImportExtractionSchema:
    def test_types_are_mapped(self):
        book, skipped = import_extraction_schema(SCHEMA)
        assert book['vocabulary']['type'] == 'category'
        assert book['vocabulary']['values'] == ['yes', 'no']
        assert book['vocabulary_focus']['type'] == 'multi_category'
        assert book['sample_size']['type'] == 'number'
        assert book['grade_level']['type'] == 'category'
        assert book['vocabulary_evidence']['type'] == 'text'
        assert skipped == ['linguistic_count']

    def test_text_fields_are_carried(self):
        book, _ = import_extraction_schema(SCHEMA)
        assert book['vocabulary']['query'] == 'Does the study examine vocabulary?'
        assert book['vocabulary']['definition'] == 'Whether vocabulary is examined'
        assert book['vocabulary']['note'] == "Matches 'vocabular'"

    def test_yaml_file(self, tmp_path):
        pytest.importorskip('yaml')
        path = tmp_path / 'extraction_schema.yml'
        path.write_text('# header\nmath_topic:\n  type: categorical\n  values: [fractions, algebra]\n',
                        encoding='utf-8')
        book, skipped = import_extraction_schema(path)
        assert book['math_topic']['values'] == ['fractions', 'algebra'] and skipped == []

    def test_non_mapping_is_rejected(self):
        with pytest.raises(ValueError, match='mapping'):
            import_extraction_schema(['vocabulary'])

    def test_missing_path_names_the_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='schema.yml'):
            import_extraction_schema(tmp_path / 'schema.yml')

    def test_empty_values_become_a_listed_problem(self):
        with pytest.raises(ValueError, match='values are required'):
            import_extraction_schema({'topic': {'type': 'categorical', 'values': None}})

    def test_unhashable_type_is_skipped(self):
        book, skipped = import_extraction_schema({'topic': {'type': ['text']},
                                                  'note': {'type': 'text'}})
        assert list(book) == ['note'] and skipped == ['topic']
