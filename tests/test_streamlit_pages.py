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
        self.assertEqual([toggle.label for toggle in app.toggle], ["Use semantic retrieval"])
        self.assertTrue(app.toggle[0].value)
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

    def test_fee_explorer_searches_are_independent(self) -> None:
        app = AppTest.from_file("pages/2_Fee_Benchmark_Explorer.py").run(timeout=60)

        app.text_input[0].set_value("cataract")
        app.button[0].click().run(timeout=60)
        app.text_input[0].set_value("endometriosis")
        app.button[0].click().run(timeout=60)

        self.assertFalse(app.exception)
        frame = app.dataframe[0].value
        descriptions = frame["benchmark"].str.lower()
        self.assertTrue(descriptions.str.contains("endometriosis").all())
        self.assertFalse(descriptions.str.contains("cataract|shoulder").any())
        self.assertFalse(frame["workbook_sheet"].str.startswith("General Principles").any())
        self.assertNotIn("fee_explorer_history", app.session_state)

        app.text_input[0].set_value("appendicitis")
        app.button[0].click().run(timeout=60)

        appendix_frame = app.dataframe[0].value
        appendix_descriptions = appendix_frame["benchmark"].str.lower()
        self.assertTrue(appendix_descriptions.str.contains("appendicectomy").all())
        self.assertFalse(appendix_descriptions.str.contains("cataract|shoulder|endometriosis").any())


if __name__ == "__main__":
    unittest.main()
