from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class StreamlitPageTests(unittest.TestCase):
    def test_all_required_pages_render_without_exceptions(self) -> None:
        expected_titles = {
            "app.py": "CareCost Navigator",
            "pages/1_Care_Pathway_Guide.py": "Care Pathway Guide",
            "pages/2_Fee_Benchmark_Explorer.py": "Fee Benchmark Explorer",
            "pages/3_About_Us.py": "About Us",
            "pages/4_Methodology.py": "Methodology",
        }
        for path, title in expected_titles.items():
            with self.subTest(path=path):
                app = AppTest.from_file(path).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertIn(title, [item.value for item in app.title])

    def test_care_pathway_retrieval_only_turn(self) -> None:
        app = AppTest.from_file("pages/1_Care_Pathway_Guide.py").run(timeout=60)
        app.chat_input[0].set_value(
            "My doctor mentioned a colonoscopy. What questions should I ask?"
        ).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("retrieval-only mode", " ".join(item.value for item in app.markdown))

    def test_fee_explorer_retrieval_only_search(self) -> None:
        app = AppTest.from_file("pages/2_Fee_Benchmark_Explorer.py").run(timeout=60)
        app.text_input[0].set_value("colonoscopy")
        app.selectbox[0].select("Day surgery / outpatient")
        app.selectbox[1].select("Doctor / professional fees")
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.dataframe), 1)
        self.assertIn("Grounded explanation", [item.value for item in app.subheader])


if __name__ == "__main__":
    unittest.main()
