import requests
import xml.etree.ElementTree as ET
import os
from config.settings import PAPER_SAVE_DIR


class ArxivClient:
    BASE_URL = "http://export.arxiv.org/api/query"

    def search_and_download(self, query, max_results=50, save_dir=None):
        save_dir = save_dir or PAPER_SAVE_DIR
        os.makedirs(save_dir, exist_ok=True)

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }

        response = requests.get(self.BASE_URL, params=params, timeout=30)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data from arXiv (status {response.status_code})")

        return self._parse_and_download(response.text, save_dir)

    def _parse_and_download(self, xml_data, save_dir):
        root = ET.fromstring(xml_data)
        papers = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            pdf_url = None

            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib["href"]

            if pdf_url:
                file_path = self._download_pdf(pdf_url, save_dir)
                papers.append({
                    "title": title,
                    "pdf_path": file_path,
                })

        return papers

    def _download_pdf(self, url, save_dir):
        file_name = url.split("/")[-1] + ".pdf"
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            return file_path

        response = requests.get(url, stream=True, timeout=60)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return file_path
