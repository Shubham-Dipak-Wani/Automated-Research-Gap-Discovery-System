import re

class SectionSegmenter:
    def segment(self, text):
        sections = {}

        patterns = [
            "abstract", "introduction", "method", "methods",
            "results", "discussion", "conclusion"
        ]

        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                sections[p] = match.start()

        # Sort sections by position
        sorted_sections = sorted(sections.items(), key=lambda x: x[1])

        segmented = {}

        for i, (name, start) in enumerate(sorted_sections):
            end = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
            segmented[name] = text[start:end]

        return segmented