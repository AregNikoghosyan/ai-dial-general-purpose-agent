import io
from pathlib import Path

import pdfplumber
import pandas as pd
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:

    def __init__(self, endpoint: str, api_key: str):
        self.client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        downloaded = self.client.files.download(file_url)
        filename = downloaded.filename
        content: bytes = downloaded.get_content()
        extension = Path(filename).suffix.lower()
        return self.__extract_text(content, extension, filename)

    def __extract_text(self, file_content: bytes, file_extension: str, filename: str) -> str:
        """Extract text content based on file type."""
        try:
            if file_extension == '.txt':
                return file_content.decode('utf-8', errors='ignore')

            if file_extension == '.pdf':
                pdf_stream = io.BytesIO(file_content)
                with pdfplumber.open(pdf_stream) as pdf:
                    pages_text = [page.extract_text() or '' for page in pdf.pages]
                return '\n'.join(pages_text)

            if file_extension == '.csv':
                decoded = file_content.decode('utf-8', errors='ignore')
                csv_buffer = io.StringIO(decoded)
                df = pd.read_csv(csv_buffer)
                return df.to_markdown(index=False)

            if file_extension in ('.html', '.htm'):
                decoded = file_content.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(decoded, 'html.parser')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                return soup.get_text(separator='\n', strip=True)

            return file_content.decode('utf-8', errors='ignore')

        except Exception as e:
            print(f"Error extracting text from file '{filename}': {e}")
            return ''
