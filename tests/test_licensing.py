import pytest

from pubextractor.licensing import RULES_DATE, RULES_VERSION, check_llm_use

CLOSED = 'no_training_no_retention'


class TestVerdicts:
    def test_abstract_allows_any_model(self):
        result = check_llm_use(content='abstract', access='personal', processing='api')
        assert result['verdict'] == 'any_llm'
        assert result['allowed'] is True

    @pytest.mark.parametrize('license_code', ['cc-by', 'CC0', 'cc-by-sa'])
    def test_open_license_allows_api(self, license_code):
        result = check_llm_use(access='open_access', license_type=license_code, processing='api')
        assert result['verdict'] == 'any_llm'
        assert result['allowed'] is True

    @pytest.mark.parametrize('license_code', ['cc-by-nc', 'cc-by-nd', 'cc-by-nc-nd'])
    def test_noncommercial_license_needs_closed_provider(self, license_code):
        standard = check_llm_use(access='open_access', license_type=license_code, processing='api')
        closed = check_llm_use(access='open_access', license_type=license_code, processing='api',
                               provider_terms=CLOSED)
        assert standard['verdict'] == 'closed_ai'
        assert standard['allowed'] is False
        assert closed['allowed'] is True

    def test_open_access_without_license_is_local_only(self):
        result = check_llm_use(access='open_access', processing='api')
        assert result['verdict'] == 'local_only'
        assert result['allowed'] is False

    def test_subscription_silent_on_ai_allows_local(self):
        result = check_llm_use(access='subscription', ai_terms='silent', processing='local')
        assert result['verdict'] == 'local_only'
        assert result['allowed'] is True

    @pytest.mark.parametrize('processing', ['api', 'private_cloud'])
    def test_subscription_silent_on_ai_refuses_off_machine(self, processing):
        result = check_llm_use(access='subscription', ai_terms='silent', processing=processing)
        assert result['allowed'] is False

    def test_subscription_allowing_ai_needs_closed_provider(self):
        standard = check_llm_use(access='subscription', ai_terms='allows', processing='api')
        closed = check_llm_use(access='subscription', ai_terms='allows', processing='api',
                               provider_terms=CLOSED)
        assert standard['verdict'] == 'closed_ai'
        assert standard['allowed'] is False
        assert closed['allowed'] is True

    def test_closed_ai_allows_private_cloud(self):
        result = check_llm_use(access='subscription', ai_terms='allows',
                               processing='private_cloud')
        assert result['allowed'] is True

    @pytest.mark.parametrize('terms', ['forbids', 'unknown'])
    def test_forbidden_or_unknown_terms_refuse_any_model(self, terms):
        result = check_llm_use(access='subscription', ai_terms=terms, processing='local')
        assert result['verdict'] == 'manual'
        assert result['allowed'] is False

    def test_personal_copy_refuses_any_model(self):
        result = check_llm_use(access='personal', processing='local')
        assert result['verdict'] == 'manual'


class TestEuropeanUnion:
    def test_research_organization_relies_on_article_3(self):
        result = check_llm_use(access='subscription', ai_terms='silent', region='eu')
        assert result['verdict'] == 'local_only'
        assert 'Art. 3' in result['reason']

    def test_other_users_face_article_4_opt_out(self):
        result = check_llm_use(access='subscription', ai_terms='silent', region='eu',
                               research_org=False)
        assert result['verdict'] == 'manual'
        assert 'Art. 4' in result['reason']


class TestReason:
    @pytest.mark.parametrize('region, phrase', [('us', 'fair use'), ('eu', 'Art. 3'),
                                                ('uk', 's29A'), ('other', 'local copyright')])
    def test_local_only_names_regional_exception(self, region, phrase):
        result = check_llm_use(access='subscription', ai_terms='silent', region=region)
        assert phrase in result['reason']

    def test_records_rules_version_date_and_inputs(self):
        result = check_llm_use(access='subscription', ai_terms='silent')
        assert result['rules_version'] == RULES_VERSION == '0.3'
        assert result['rules_date'] == RULES_DATE
        assert result['inputs']['access'] == 'subscription'
        assert result['inputs']['provider_terms'] == 'standard'


class TestValidation:
    @pytest.mark.parametrize('field', ['content', 'access', 'ai_terms', 'processing', 'region',
                                       'provider_terms'])
    def test_rejects_unknown_value(self, field):
        with pytest.raises(ValueError, match=field):
            check_llm_use(**{field: 'nonsense'})


class TestRulesVersion03:
    @pytest.mark.parametrize('access', ['open_access', 'subscription', 'personal'])
    def test_license_goes_with_the_work(self, access):
        result = check_llm_use(access=access, license_type='cc-by', processing='api')
        assert result['verdict'] == 'any_llm'

    def test_open_access_terms_forbidding_ai_are_respected(self):
        result = check_llm_use(access='open_access', ai_terms='forbids')
        assert result['verdict'] == 'manual'

    def test_open_access_terms_allowing_ai_are_respected(self):
        result = check_llm_use(access='open_access', ai_terms='allows', processing='api',
                               provider_terms=CLOSED)
        assert result['verdict'] == 'closed_ai'
        assert result['allowed'] is True

    def test_abstract_under_terms_forbidding_ai(self):
        result = check_llm_use(content='abstract', ai_terms='forbids', processing='api')
        assert result['verdict'] == 'closed_ai'
        assert result['allowed'] is False

    def test_uk_commercial_work_is_referred(self):
        result = check_llm_use(access='subscription', ai_terms='silent', region='uk',
                               research_org=False)
        assert result['verdict'] == 'manual'
        assert 's29A' in result['reason']

    @pytest.mark.parametrize('region, phrase', [('uk', 's29A(5)'), ('eu', 'Art. 7(1)'),
                                                ('us', 'obtain permission.')])
    def test_forbidding_terms_note_unenforceability_where_it_applies(self, region, phrase):
        result = check_llm_use(access='subscription', ai_terms='forbids', region=region)
        assert result['reason'].endswith(phrase) or phrase in result['reason']

    def test_reasons_are_accurate(self):
        cc0 = check_llm_use(access='open_access', license_type='cc0', processing='api')
        us = check_llm_use(access='subscription', ai_terms='silent', region='us')
        assert 'where required' in cc0['reason']
        assert '107' in us['reason'] and 'case by case' in us['reason']
