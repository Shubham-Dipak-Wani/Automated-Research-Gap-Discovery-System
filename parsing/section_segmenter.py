import re


class SectionSegmenter:
    """Split paper text into named sections using common heading names.

    A heading must sit on its own line (optionally numbered, like "3. Results").
    If no heading is found, the whole text is returned as one "body" section
    instead of being dropped.
    """

    PATTERNS = [
        "abstract", "introduction", "background", "related work",
        "method", "methods", "methodology", "approach",
        "experiments", "evaluation", "results", "discussion",
        "limitations", "future work", "conclusion", "conclusions",
    ]

    def segment(self, text):
        starts = {}
        for name in self.PATTERNS:
            pattern = rf"^\s*(?:\d+(?:\.\d+)*\.?\s*)?{re.escape(name)}\s*$"
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                starts[name] = match.start()

        if not starts:
            return {"body": text} if text.strip() else {}

        ordered = sorted(starts.items(), key=lambda x: x[1])
        sections = {}
        for i, (name, start) in enumerate(ordered):
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
            sections[name] = text[start:end]
        return sections
