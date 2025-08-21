import logging

from tokenizer.tokenizer import SimpleTokenizerV1, download_file, tokenize_text

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.DEBUG)
    _log.info("Starting the download process...")
    file_path = "resources/the-verdict.txt"
    text: str = download_file(file_path)
    _log.info("Download process completed.")

    tokenized_text = tokenize_text(text)
    _log.info(f"Tokenized text: {tokenized_text[:10]}")
    all_words = sorted(set(tokenized_text))
    vocab_size = len(all_words)
    _log.info(f"Vocabulary size: {vocab_size}")
    text = (
        "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."
    )

    tokenizer = SimpleTokenizerV1(vocab={tkn: i for i, tkn in enumerate(all_words)})
    ids = tokenizer.encode(text)
    _log.info(f"Encoded text: {ids}")
    decoded_text = tokenizer.decode(ids)
    _log.info(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
