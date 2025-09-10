import logging

import tiktoken
from torch.utils.data import DataLoader

from gpt.dataset import GPTDatasetV1
from tokenizer.tokenizer import download_file

_log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    _log.info("Starting the download process...")
    file_path = "resources/the-verdict.txt"
    text: str = download_file(file_path)
    dlod: DataLoader[str] = create_dataloader_v1(
        text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dlod)
    first_batch = next(data_iter)
    _log.info(f"First batch: {first_batch}")
    second_batch = next(data_iter)
    _log.info(f"Second batch: {second_batch}")


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader[str]:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    main()
