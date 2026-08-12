from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.mesh_rag import MeshIndex, MeshRecord, _rank_records_for_candidates, generate_query_candidates


class MeshRetrievalTests(unittest.TestCase):
    def test_candidate_ranking_recovers_clinical_term_from_fee_form_boilerplate(self) -> None:
        records = [
            MeshRecord("D013270", "Stomach", ("gastric",)),
            MeshRecord("D010146", "Pain", ()),
        ]
        candidates = generate_query_candidates(
            "Estimate the fee benchmark for stomach pain after eating. Care setting: Not sure."
        )
        matches = _rank_records_for_candidates(candidates, records, limit=5, min_score=0.45)

        terms = {match.record.preferred_term.lower() for match in matches}
        self.assertIn("stomach", terms)
        self.assertIn("pain", terms)

    def test_index_keeps_top_ranked_clinical_match_at_configured_threshold(self) -> None:
        record = MeshRecord("D013270", "Stomach", ("gastric",))
        with patch("utils.mesh_rag._fetch_mesh_candidates_batch", return_value=(record,)):
            matches = MeshIndex().search("What might stomach pain after eating cost?", min_score=0.45)

        self.assertEqual(matches[0].record.ui, "D013270")
        self.assertGreaterEqual(matches[0].score, 0.45)


if __name__ == "__main__":
    unittest.main()
