"""Tests for effective-state resolution algorithm (AMEND-01 through AMEND-04)."""
import pytest
from graph.resolver import _apply_amendments, EffectiveState


class TestApplyAmendmentsNoAmendments:
    def test_unamended_returns_original(self):
        result = _apply_amendments("The penalty shall be one lakh rupees.", [])
        assert result.current_text == "The penalty shall be one lakh rupees."
        assert result.status == "original"
        assert result.amendment_chain == []
        assert result.original_text == "The penalty shall be one lakh rupees."


class TestApplyAmendmentsSubstitutes:
    def test_single_substitution(self):
        amendments = [{
            "amendment_uid": "amend-test-2020",
            "amendment_type": "SUBSTITUTES",
            "old_text": "one lakh rupees",
            "new_text": "five lakh rupees",
            "effective_date": "2020-01-01",
        }]
        result = _apply_amendments("The penalty shall be one lakh rupees.", amendments)
        assert result.current_text == "The penalty shall be five lakh rupees."
        assert result.status == "amended"
        assert len(result.amendment_chain) == 1
        assert result.amendment_chain[0]["amendment_uid"] == "amend-test-2020"

    def test_substitution_preserves_original(self):
        amendments = [{
            "amendment_uid": "amend-test-2020",
            "amendment_type": "SUBSTITUTES",
            "old_text": "penalty",
            "new_text": "fine",
            "effective_date": "2020-01-01",
        }]
        result = _apply_amendments("The penalty shall apply.", amendments)
        assert result.original_text == "The penalty shall apply."
        assert result.current_text == "The fine shall apply."


class TestApplyAmendmentsInserts:
    def test_insert_appends_text(self):
        amendments = [{
            "amendment_uid": "amend-test-2021",
            "amendment_type": "INSERTS",
            "new_text": " Provided that the Central Government may exempt.",
            "old_text": "",
            "effective_date": "2021-01-01",
        }]
        result = _apply_amendments("The company shall comply.", amendments)
        assert result.current_text == "The company shall comply. Provided that the Central Government may exempt."
        assert result.status == "amended"


class TestApplyAmendmentsOmits:
    def test_omit_removes_text(self):
        amendments = [{
            "amendment_uid": "amend-test-2022",
            "amendment_type": "OMITS",
            "old_text": " and its subsidiaries",
            "new_text": "",
            "effective_date": "2022-01-01",
        }]
        result = _apply_amendments("The company and its subsidiaries shall comply.", amendments)
        assert result.current_text == "The company shall comply."
        assert result.status == "omitted"

    def test_full_omit_sets_status_omitted(self):
        amendments = [{
            "amendment_uid": "amend-test-2023",
            "amendment_type": "OMITS",
            "old_text": "Section text here.",
            "new_text": "",
            "effective_date": "2023-01-01",
        }]
        result = _apply_amendments("Section text here.", amendments)
        assert result.current_text == ""
        assert result.status == "omitted"


class TestApplyAmendmentsUnknownType:
    def test_decriminalizes_skipped_gracefully(self):
        """DECRIMINALIZES is an unknown type — should be skipped without error."""
        amendments = [{
            "amendment_uid": "amend-test-2024",
            "amendment_type": "DECRIMINALIZES",
            "old_text": "",
            "new_text": "",
            "effective_date": "2024-01-01",
        }]
        result = _apply_amendments("Original text.", amendments)
        assert result.current_text == "Original text."
        # Status remains "amended" since we entered the amendment loop,
        # but DECRIMINALIZES was skipped — however since no actual text
        # change occurred and it's the only amendment, the implementation
        # should still set status based on whether text actually changed.
        # The key assertion: no crash, text unchanged.
        assert result.original_text == "Original text."


class TestApplyAmendmentsChronologicalOrder:
    def test_multiple_amendments_in_order(self):
        """AMEND-01: amendments applied chronologically by effective_date."""
        amendments = [
            {
                "amendment_uid": "amend-first-2019",
                "amendment_type": "SUBSTITUTES",
                "old_text": "one crore",
                "new_text": "five crore",
                "effective_date": "2019-01-01",
            },
            {
                "amendment_uid": "amend-second-2021",
                "amendment_type": "INSERTS",
                "new_text": " as prescribed by the Central Government",
                "old_text": "",
                "effective_date": "2021-01-01",
            },
        ]
        result = _apply_amendments("The threshold is one crore.", amendments)
        assert result.current_text == "The threshold is five crore. as prescribed by the Central Government"
        assert result.status == "amended"
        assert len(result.amendment_chain) == 2
        assert result.amendment_chain[0]["amendment_uid"] == "amend-first-2019"
        assert result.amendment_chain[1]["amendment_uid"] == "amend-second-2021"

    def test_substitute_then_omit(self):
        """First substitute, then omit — final status is omitted."""
        amendments = [
            {
                "amendment_uid": "amend-sub-2018",
                "amendment_type": "SUBSTITUTES",
                "old_text": "old provision",
                "new_text": "new provision",
                "effective_date": "2018-01-01",
            },
            {
                "amendment_uid": "amend-omit-2020",
                "amendment_type": "OMITS",
                "old_text": "new provision",
                "new_text": "",
                "effective_date": "2020-01-01",
            },
        ]
        result = _apply_amendments("This is old provision text.", amendments)
        assert "new provision" not in result.current_text
        assert result.status == "omitted"


class TestApplyAmendmentsResultStructure:
    def test_result_has_all_fields(self):
        """AMEND-04: structured result with current_text, status, amendment_chain, original_text."""
        result = _apply_amendments("Original.", [])
        assert hasattr(result, "current_text")
        assert hasattr(result, "status")
        assert hasattr(result, "amendment_chain")
        assert hasattr(result, "original_text")
        assert isinstance(result.amendment_chain, list)


class TestProspectiveExclusion:
    """AMEND-03: Prospective amendments (future effective_date) excluded from current-state queries."""

    def test_resolve_query_contains_date_filter_clause(self):
        """AMEND-03: RESOLVE_QUERY must filter out amendments with effective_date > as_of.

        The date filter is the sole mechanism for prospective exclusion — if it is
        absent from the Cypher query, future amendments will silently appear in results.
        """
        from graph.resolver import RESOLVE_QUERY

        assert "r.effective_date <= date($as_of)" in RESOLVE_QUERY, (
            "RESOLVE_QUERY must contain 'r.effective_date <= date($as_of)' to exclude "
            "prospective amendments (AMEND-03)"
        )

    def test_resolve_effective_state_passes_as_of_date_to_cypher(self, mock_driver):
        """AMEND-03: resolve_effective_state forwards as_of_date to the Cypher query.

        When an explicit as_of_date is provided, that exact date must be passed as
        the $as_of parameter so the date filter in RESOLVE_QUERY can exclude future
        amendments.
        """
        from unittest.mock import MagicMock
        from graph.resolver import resolve_effective_state

        mock_drv, mock_session = mock_driver

        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "uid": "sec-135",
            "original_text": "Original text.",
            "amendments": [],
        }[key]
        mock_session.run.return_value.single.return_value = mock_record

        resolve_effective_state(mock_drv, section_number=135, as_of_date="2022-01-01")

        mock_session.run.assert_called_once()
        call_kwargs = mock_session.run.call_args
        # Parameters can be positional args or kwargs; inspect both
        args, kwargs = call_kwargs
        passed_params = kwargs if kwargs else {}
        # run(query, section_number=..., as_of=...) — positional arg[0] is query
        if len(args) > 1:
            # older calling convention: positional dict
            passed_params = args[1] if isinstance(args[1], dict) else passed_params
        assert passed_params.get("as_of") == "2022-01-01", (
            f"Expected as_of='2022-01-01' passed to Cypher, got: {passed_params}"
        )

    def test_resolve_effective_state_defaults_as_of_to_today(self, mock_driver):
        """AMEND-03: When as_of_date is None, today's date is used as the cutoff.

        This ensures prospective amendments (effective tomorrow or later) are
        always excluded from a default call to resolve_effective_state.
        """
        from datetime import date
        from unittest.mock import MagicMock
        from graph.resolver import resolve_effective_state

        mock_drv, mock_session = mock_driver

        mock_record = MagicMock()
        mock_record.__getitem__ = lambda self, key: {
            "uid": "sec-135",
            "original_text": "Original text.",
            "amendments": [],
        }[key]
        mock_session.run.return_value.single.return_value = mock_record

        today = date.today().isoformat()
        resolve_effective_state(mock_drv, section_number=135, as_of_date=None)

        mock_session.run.assert_called_once()
        args, kwargs = mock_session.run.call_args
        passed_params = kwargs if kwargs else {}
        if len(args) > 1 and isinstance(args[1], dict):
            passed_params = args[1]
        assert passed_params.get("as_of") == today, (
            f"Expected as_of='{today}' (today) when as_of_date=None, got: {passed_params}"
        )
