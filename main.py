import logging

import tiktoken

from gpt.dataset import GPTDatasetV1
from tokenizer.tokenizer import download_file

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG)
    _log.info("Starting the download process...")
    file_path = "resources/the-verdict.txt"
    text: str = download_file(file_path)
    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(text)
    _log.info(f"Encoded text: {len(enc_text)}...")
    enc_sample = enc_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1 : context_size + 1]
    _log.info(f"(x): {x}")
    _log.info(f"(y): {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        _log.info(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")

    ds = GPTDatasetV1()


if __name__ == "__main__":
    main()
