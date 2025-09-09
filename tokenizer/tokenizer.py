import logging
import os
import re
import urllib.request
from typing import Any

_log: logging.Logger = logging.getLogger(__name__)


def download_file(destination: str) -> str:
    """Download a file from a URL to a specified destination."""
    if not os.path.exists(destination):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        _log.info(f"Downloading {url} to {destination}")
        try:
            urllib.request.urlretrieve(url, destination)
            _log.info(f"Downloaded {url} to {destination}")
        except Exception as e:
            _log.error(f"Failed to download {url} to {destination}: {e}")
            raise
    else:
        _log.info(f"File {destination} already exists. Skipping download.")

    with open(destination, "r", encoding="utf-8") as file:
        return file.read()


def tokenize_text(text: str) -> list[str]:
    """Tokenize the input text into words."""
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in result if item.strip()]


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, Any]) -> None:
        self.str_to_int: dict[str, int] = vocab
        self.int_to_str: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Convert text to a list of integers based on the vocabulary."""
        preprocessed: list[str | Any] = re.split(
            pattern=r'([,.:;?_!"()\']|--|\s)', string=text
        )
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integers back to text."""
        text: str = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, Any]) -> None:
        self.str_to_int: dict[str, int] = vocab
        self.int_to_str: dict[int, str] = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed: list[str | Any] = re.split(
            pattern=r'([,.:;?_!"()\']|--|\s)', string=text
        )
        preprocessed = [
            item if item in preprocessed else "<|unk|>" for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integers back to text."""
        text: str = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
