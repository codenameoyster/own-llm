import logging
import os
import urllib.request

_log = logging.getLogger(__name__)


def download_file(destination: str) -> str:
    """Download a file from a URL to a specified destination."""
    if not os.path.exists(destination):
        url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
        _log.info(f"Downloading {url} to {destination}")
        try:
            urllib.request.urlretrieve(url, destination)
            _log.info(f"Downloaded {url} to {destination}")
        except Exception as e:
            _log.error(f"Failed to download {url} to {destination}: {e}")
            raise
    else:
        _log.info(f"File {destination} already exists. Skipping download.")

    with open(destination, 'r', encoding='utf-8') as file:
        return file.read()


def tokenize_text(text: str) -> list:
    """Tokenize the input text into words."""
    import re
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]

class SimpleTokenizerV1:
    def __init__(self, vocab: dict):
        self.str_to_init: dict = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    def encode(self, text: str) -> list:
        """Convert text to a list of integers based on the vocabulary."""
        return []
