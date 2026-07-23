from __future__ import annotations

import unittest

from llm.openai import _extract_json_object, _parse_json_args


class OpenAIFallbackTest(unittest.TestCase):
    def test_extract_json_object_from_markdown_or_preface(self) -> None:
        payload = """
        Sure, here is the result:

        ```json
        {"needs_revision": false, "reasoning": "Looks good."}
        ```
        """
        extracted = _extract_json_object(payload)
        parsed = _parse_json_args(extracted)
        self.assertEqual(parsed["needs_revision"], False)
        self.assertEqual(parsed["reasoning"], "Looks good.")


if __name__ == "__main__":
    unittest.main()
