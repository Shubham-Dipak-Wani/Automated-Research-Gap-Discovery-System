import requests
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

class ArxivClient:
    BASE_URL = "http://export.arxiv.org/api/query"

    def search_and_download(self, query, max_results=5, save_dir="data/raw_papers"):
        os.makedirs(save_dir, exist_ok=True)

        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results
        }

        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise Exception("Failed to fetch data from arXiv")

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
                    "pdf_path": file_path
                })

        return papers

    def _download_pdf(self, url, save_dir):
        file_name = url.split("/")[-1] + ".pdf"
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            return file_path

        response = requests.get(url, stream=True)

        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        return file_path