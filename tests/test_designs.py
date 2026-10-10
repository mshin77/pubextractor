import pytest

from pubextractor import DESIGN_TYPES, suggest_assignment, suggest_design, suggest_scd_design


def _pages(*texts):
    return [{'page': i + 1, 'text': text} for i, text in enumerate(texts)]


SCD_METHOD = ('Method\nParticipants\nFour students took part.\nDesign\nA multiple baseline '
              'design across participants was used. A pretest screened each student.\n'
              'Results\nAll students gained.')
GROUP_METHOD = ('Method\nParticipants\nClasses were randomly assigned to the treatment or the '
                'control group. Pretest scores entered an ANCOVA.\nResults\nScores rose.')


class TestSuggestDesign:
    def test_single_case_terms_in_method(self):
        design, evidence = suggest_design(_pages(SCD_METHOD))
        assert design == 'single_case'
        assert any('multiple baseline' in item for item in evidence)

    def test_group_terms_in_method(self):
        design, evidence = suggest_design(_pages(GROUP_METHOD))
        assert design == 'group'
        assert any('randomly assigned' in item for item in evidence)

    def test_review_of_other_designs_before_method_is_ignored(self):
        intro = ('Introduction\nEarlier single-subject studies used a reversal design and an '
                 'ABAB design with single-case graphs.\n')
        assert suggest_design(_pages(intro + GROUP_METHOD))[0] == 'group'

    def test_references_are_ignored_without_a_method_heading(self):
        text = ('A multiple-probe design across three students was used.\nReferences\n'
                'Smith (2001). A quasi-experimental study with a comparison group.')
        assert suggest_design(_pages(text))[0] == 'single_case'

    def test_correlation_table_without_design_terms(self):
        grid = {'table': 2, 'caption': 'Table 2. Correlations among measures',
                'rows': [['', '1', '2'], ['1. Reading', '-', ''], ['2. Maths', '.42', '-']]}
        design, evidence = suggest_design(_pages('Method\nSurvey of 300 students.'), [grid])
        assert design == 'correlational'
        assert evidence == ['correlation table 2']

    def test_interview_and_theme_words(self):
        text = ('Method\nSemi-structured interviews with ten teachers were coded; three themes '
                'emerged from the thematic analysis.')
        assert suggest_design(_pages(text))[0] == 'qualitative'

    def test_single_case_figure_caption_outweighs_a_pretest(self):
        figures = [{'figure': 1, 'caption': 'Figure 1. Correct responses across sessions'}]
        design, evidence = suggest_design(_pages('Method\nStudents took a pretest.'),
                                          figures=figures)
        assert design == 'single_case'
        assert evidence == ['figure 1 caption']

    @pytest.mark.parametrize('text, expected', [
        ('Method\nClasses were randomly assigned to conditions.', ('rct', ['randomly assigned (1)'])),
        ('Method\nA quasi-experimental design with matched classes.',
         ('qed', ['quasi-experimental (1)', 'matched (1)'])),
        ('Method\nStudents took a pretest.', ('rct', [])),
        ('Method\nA quasi-experimental, non-randomized design.',
         ('qed', ['quasi-experimental (1)', 'nonrandom (1)'])),
        ('Method\nIntact classes with nonrandom assignment.',
         ('qed', ['intact classes (1)', 'nonrandom (1)'])),
        ('Method\nClasses were not randomized.', ('qed', ['not randomized (1)'])),
        ('Method\nClasses were not randomly assigned.', ('qed', ['not randomly assigned (1)'])),
        ('Method\nClasses were randomly assigned and matched.',
         ('rct', ['tie: randomly assigned (1), matched (1)']))])
    def test_assignment_suggestion(self, text, expected):
        assert suggest_assignment(_pages(text)) == expected

    @pytest.mark.parametrize('text, expected', [
        ('Method\nA multiple-baseline design across three students.',
         ('multiple_baseline', ['multiple baseline (1)'])),
        ('Method\nAn ABAB withdrawal design was used.', ('reversal', ['reversal (2)'])),
        ('Method\nFour students took part.', ('ab', []))])
    def test_scd_design_suggestion(self, text, expected):
        assert suggest_scd_design(_pages(text)) == expected

    def test_figure_caption_lines_feed_the_design(self):
        from pubextractor.designs import figure_captions
        pages = _pages('Method\nStudents took a pretest.\nFigure 1. Correct responses across sessions')
        figures = figure_captions(pages)
        assert figures == [{'figure': 1, 'caption': 'Figure 1. Correct responses across sessions'}]
        assert suggest_design(pages, figures=figures) == ('single_case', ['figure 1 caption'])

    def test_tie_goes_to_single_case(self):
        text = 'Method\nA reversal design was used; the control group condition was withdrawn.'
        assert suggest_design(_pages(text)) == ('single_case', ['reversal design (1)'])

    def test_blank_header_cell_is_read(self):
        grid = {'table': 1, 'caption': '', 'rows': [[None, 'Correlation'], ['a', '.3']]}
        assert suggest_design(_pages('Method\nSurvey.'), [grid])[0] == 'correlational'

    @pytest.mark.parametrize('pages', [[], _pages('')])
    def test_no_terms_falls_back_to_group(self, pages):
        assert suggest_design(pages) == ('group', ['no design terms found'])

    def test_every_suggestion_is_a_design_type(self):
        assert DESIGN_TYPES == ('group', 'single_case', 'correlational', 'qualitative')


@pytest.mark.parametrize('text, caption, expected', [
    ('Method\nDisruptive behavior was recorded each session.', '', ('decrease', ['disruptive (1)'])),
    ('Method\nWords read correctly.', 'Figure 1. Self-injury per session',
     ('decrease', ['self-injury (1)'])),
    ('Method\nWords read correctly.', '', ('increase', [])),
    ('Method\nStudents receiving free or reduced-price lunch read words.', '', ('increase', []))])
def test_goal_suggestion(text, caption, expected):
    from pubextractor import suggest_goal
    assert suggest_goal(_pages(text), caption) == expected
