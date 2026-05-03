from parsing.pdf_parser import PDFParser, strip_references
from parsing.section_segmenter import SectionSegmenter


def test_references_mentioned_mid_sentence_do_not_truncate_the_paper():
    text = (
        "Abstract\nWe study retrieval with references to prior work in the introduction.\n"
        "Results\nRetrieval improves accuracy by 12%.\n"
        "References\n[1] Lewis et al. 2020."
    )
    cleaned = PDFParser()._clean_text(text)
    assert "Retrieval improves accuracy by 12%." in cleaned
    assert "Lewis et al." not in cleaned


def test_numbered_and_uppercase_reference_headings():
    assert strip_references("Body text.\n7 REFERENCES\n[1] A.").strip() == "Body text."
    assert strip_references("Body.\nBibliography\nB.").strip() == "Body."


def test_last_references_heading_wins():
    text = "Intro\nReferences\nnot really\nMain results here.\nReferences\n[1] Real bib."
    out = strip_references(text)
    assert "Main results here." in out and "Real bib" not in out


def test_text_without_references_is_unchanged():
    assert strip_references("Just a body.") == "Just a body."


def test_whitespace_is_normalized_but_paragraphs_kept():
    cleaned = PDFParser()._clean_text("A  lot\tof   space.\n\n\n\nNew paragraph.")
    assert cleaned == "A lot of space.\n\nNew paragraph."


def test_segmenter_finds_numbered_headings_in_order():
    text = "Abstract\nShort summary.\n1 Introduction\nWhy.\n3.2 Results\nNumbers.\nLimitations\nCaveats."
    sections = SectionSegmenter().segment(text)
    assert list(sections) == ["abstract", "introduction", "results", "limitations"]
    assert "Numbers." in sections["results"] and "Caveats." not in sections["results"]


def test_heading_words_inside_sentences_are_not_headings():
    sections = SectionSegmenter().segment("Abstract\nThe introduction of RAG changed results in QA.")
    assert list(sections) == ["abstract"]


def test_paper_without_headings_is_kept_as_body():
    """Previously such a paper produced no sections, so none of its sentences were used."""
    assert SectionSegmenter().segment("A paper with no headings at all.") == {"body": "A paper with no headings at all."}
    assert SectionSegmenter().segment("   ") == {}
