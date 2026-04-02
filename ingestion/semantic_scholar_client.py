import requests
import os
import time
from config.settings import SEMANTIC_SCHOLAR_API_URL, PAPER_SAVE_DIR


class SemanticScholarClient:
    """Supplements ArXiv papers with metadata and citation graph discovery."""

    def enrich_and_expand(self, papers, max_extra_papers=10):
        """
        For each paper, fetch metadata from Semantic Scholar.
        Use citation graph to discover related papers.
        Returns enriched paper list + newly discovered papers.
        """
        enriched = []
        seen_titles = {p["title"].lower() for p in papers}
        discovered = []

        for paper in papers:
            metadata = self._fetch_metadata(paper["title"])
            if metadata:
                paper["citation_count"] = metadata.get("citationCount", 0)
                paper["year"] = metadata.get("year")
                paper["venue"] = metadata.get("venue", "")
                paper["semantic_scholar_id"] = metadata.get("paperId", "")

                # Discover related papers from references
                refs = self._fetch_references(metadata["paperId"])
                for ref in refs:
                    ref_title = ref.get("title", "").lower()
                    if ref_title and ref_title not in seen_titles:
                        seen_titles.add(ref_title)
                        discovered.append(ref)

            enriched.append(paper)
            time.sleep(0.5)  # Rate limiting

        # Download top cited discovered papers
        extra = self._download_discovered(discovered, max_extra_papers)

        return enriched, extra

    def _fetch_metadata(self, title):
        """Search Semantic Scholar by title."""
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/search",
                params={"query": title, "limit": 1,
                        "fields": "paperId,title,citationCount,year,venue"},
                timeout=10,
            )
            if response.status_code == 200:
                results = response.json().get("data", [])
                return results[0] if results else None
        except requests.RequestException:
            pass
        return None

    def _fetch_references(self, paper_id):
        """Get references for a paper."""
        try:
            response = requests.get(
                f"{SEMANTIC_SCHOLAR_API_URL}/paper/{paper_id}/references",
                params={"fields": "title,citationCount,year,externalIds", "limit": 20},
                timeout=10,
            )
            if response.status_code == 200:
                refs = response.json().get("data", [])
                return [r["citedPaper"] for r in refs if r.get("citedPaper")]
        except requests.RequestException:
            pass
        return []

    def _download_discovered(self, discovered, max_papers):
        """Download PDFs for top-cited discovered papers that have ArXiv IDs."""
        # Sort by citation count, take top N
        with_citations = [d for d in discovered if d.get("citationCount")]
        with_citations.sort(key=lambda x: x.get("citationCount", 0), reverse=True)

        extra_papers = []
        for paper in with_citations[:max_papers]:
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            if not arxiv_id:
                continue

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            file_name = f"{arxiv_id}.pdf"
            file_path = os.path.join(PAPER_SAVE_DIR, file_name)

            if not os.path.exists(file_path):
                try:
                    resp = requests.get(pdf_url, stream=True, timeout=30)
                    if resp.status_code == 200:
                        os.makedirs(PAPER_SAVE_DIR, exist_ok=True)
                        with open(file_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024):
                                if chunk:
                                    f.write(chunk)
                    else:
                        continue
                except requests.RequestException:
                    continue

            extra_papers.append({
                "title": paper.get("title", "Unknown"),
                "pdf_path": file_path,
                "citation_count": paper.get("citationCount", 0),
                "year": paper.get("year"),
            })
            time.sleep(1)  # Rate limiting for downloads

        return extra_papers
