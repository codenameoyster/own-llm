
import tiktoken
import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset[str]):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []
        token_ids: list[int] = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk: list[int] = token_ids[i : i + max_length]
            target_chunk: list[int] = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        return None

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]
