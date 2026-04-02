import re


class SectionSegmenter:
    PATTERNS = [
        "abstract", "introduction", "method", "methods",
        "methodology", "results", "discussion", "conclusion",
        "related work", "background", "evaluation", "experiments"
    ]

    def segment(self, text):
        sections = {}

        for p in self.PATTERNS:
            # Match section headers at line starts, with optional numbering
            pattern = rf'^\s*(?:\d+\.?\s*)?{p}\s*$'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                sections[p] = match.start()

        sorted_sections = sorted(sections.items(), key=lambda x: x[1])

        segmented = {}

        for i, (name, start) in enumerate(sorted_sections):
            end = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
            segmented[name] = text[start:end]

        return segmented
