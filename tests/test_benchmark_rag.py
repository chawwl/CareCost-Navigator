from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.benchmark_rag import (
    build_benchmark_index,
    expand_query_terms,
    has_specific_match,
    is_benchmark_evidence_record,
    is_guidance_record,
    make_record,
    search_benchmark_records,
    specific_term_score,
)


class BenchmarkRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endometriosis = make_record(
            "Surg & Ana Single TOSP FB",
            10,
            {
                "tosp": "SI715U",
                "description": "Uterus, MIS Ablation Of Endometriosis (Simple)",
                "lower_bound": "4500",
                "upper_bound": "7400",
            },
        )
        self.appendicectomy = make_record(
            "Surg Hospital Single TOSP FB",
            11,
            {
                "tosp": "SF849A",
                "description": "Appendix, Various Lesions, Appendicectomy Without Drainage",
                "lower_bound": "10200",
                "upper_bound": "14600",
            },
        )
        self.cataract = make_record(
            "Surg Hospital Single TOSP FB",
            12,
            {
                "tosp": "SL808L",
                "description": "Lens, Cataract, Extraction With Intra-Ocular Lens Implant",
                "lower_bound": "2600",
                "upper_bound": "4700",
                "explanatory_notes": "Note: most cases involve an implant.",
            },
        )
        self.shoulder = make_record(
            "Surg Hospital Single TOSP FB",
            13,
            {
                "tosp": "SB710S",
                "description": "Shoulder Soft Tissue Injury, Decompression With Cuff Repair",
                "lower_bound": "13800",
                "upper_bound": "18800",
                "explanatory_notes": "Note: most cases involve an implant.",
            },
        )
        self.guidance = make_record(
            "General Principles- Surg&Ana FB",
            14,
            {"note": "Appendicectomy fee benchmarks are references and are not fee caps."},
        )
        self.index = build_benchmark_index(
            [
                self.endometriosis,
                self.appendicectomy,
                self.cataract,
                self.shoulder,
                self.guidance,
            ]
        )

    def test_specific_matching_uses_whole_tokens(self) -> None:
        self.assertFalse(has_specific_match("general principles note", ["not"], []))
        self.assertEqual(specific_term_score("general principles note", ["not"]), 0.0)

    def test_appendicitis_expands_to_workbook_procedure_terms(self) -> None:
        expanded = set(expand_query_terms(["appendicitis"]))
        self.assertTrue({"appendicectomy", "appendectomy"} <= expanded)

    def test_form_boilerplate_does_not_override_clinical_anchor(self) -> None:
        query = (
            "Estimate the fee benchmark for endometriosis. "
            "Care setting: Not sure. Fee component: Both hospital and doctor fees. "
            "Additional context: none."
        )

        matches = search_benchmark_records(self.index, query, "Procedure cost estimate")

        self.assertEqual([record.record_id for record, _ in matches], [self.endometriosis.record_id])

    def test_appendicitis_returns_appendicectomy_and_excludes_guidance(self) -> None:
        query = (
            "Estimate the fee benchmark for appendicitis. "
            "Care setting: Not sure. Fee component: Both hospital and doctor fees. "
            "Additional context: none."
        )

        matches = search_benchmark_records(self.index, query, "Procedure cost estimate")

        self.assertEqual([record.record_id for record, _ in matches], [self.appendicectomy.record_id])
        self.assertTrue(all(not is_guidance_record(record) for record, _ in matches))

    def test_guidance_classification_does_not_depend_on_parsed_amounts(self) -> None:
        attendance = make_record("Inpatient Attendance FB", 3, {"note": "Office hours: S$210 to S$350"})
        continuation = make_record("Surgeon & Ana Multiple TOSP FB", 4, {"description": "Second procedure"})

        self.assertTrue(is_guidance_record(self.guidance))
        self.assertFalse(is_benchmark_evidence_record(self.guidance))
        self.assertTrue(is_benchmark_evidence_record(attendance))
        self.assertTrue(is_benchmark_evidence_record(continuation))

    def test_generic_policy_search_can_still_return_guidance(self) -> None:
        matches = search_benchmark_records(
            self.index,
            "What are fee benchmarks?",
            "Procedure cost estimate",
        )

        self.assertIn(self.guidance.record_id, {record.record_id for record, _ in matches})

    def test_duplicate_query_variants_do_not_multiply_scores(self) -> None:
        with patch(
            "utils.benchmark_rag.build_multi_query_variants",
            return_value=["endometriosis"],
        ):
            baseline = search_benchmark_records(
                self.index,
                "endometriosis",
                "Procedure cost estimate",
                limit=1,
            )

        with patch(
            "utils.benchmark_rag.build_multi_query_variants",
            return_value=["endometriosis"] * 4,
        ):
            duplicated = search_benchmark_records(
                self.index,
                "endometriosis",
                "Procedure cost estimate",
                limit=1,
            )

        self.assertEqual(baseline[0][0].record_id, duplicated[0][0].record_id)
        self.assertAlmostEqual(baseline[0][1], duplicated[0][1], places=6)


if __name__ == "__main__":
    unittest.main()
