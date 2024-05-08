from dataclasses import dataclass
import enum
import os
from pathlib import Path
from typing import Iterator, Sequence

from sentencepiece import SentencePieceProcessor
import torch


class Modality(enum.IntEnum):

    """Chunk modality type.

    MIXED: preserve whitespace (like source code), but do not insert any control chars.
    NATURAL: normalize aggressively, collapse whitespaces. You usually do not want this.
    CODE:    preserve whitespace, insert the file header and related control chars.
    CONTROL: custom control character.
    """

    MIXED = 0
    NATURAL = 1
    CODE = 2
    CONTROL = 3


@dataclass(frozen=True, slots=True)
class Chunk:

    """Part of the input.

    `code_file_path` is the source code file path, e.g., on GitHub. Example:
    `poolside/monster/bone.py`. It's nice to provide it, but not strictly required. It is ignored
    in `MIXED` and `NATURAL` modalities.
    `code_line_start` sets the starting line number. If None (default), line numbers are not
    inserted. If zero, line numbers are already inserted into `contents`, but must be divided
    with a single whitespace from the actual line contents.
    """

    contents: str | bytes
    mod: Modality
    code_file_path: str = ""
    code_line_start: int | None = None


class Tokenizer:

    """Text tokenizer.

    We distinguish two modalities: natural language and source code. They obey to different,
    optimized tokenization rules.
    """

    def __init__(self, model_path: str | Path):
        """Initialize a new instance of the Tokenizer class.

        :param model_path: Path to the trained SentencePiece BPE model.
        """
        self._model = SentencePieceProcessor(os.path.expanduser(str(model_path)))
        self._verbatim = b"\x01"
        self._ps_code_start = self._model.bos_id()
        self._first_control = self._ps_code_end = self._model.PieceToId("<0x00>")
        self._ps_code_meta_start = self._model.PieceToId("<0x01>")
        self._ps_code_meta_end = self._model.PieceToId("<0x02>")
        self._ps_line_delim = self._model.PieceToId("<0x06>")
        self._ps_xcontrol_start = self._model.PieceToId("<0x07>")
        self._ps_xcontrol_end = self._model.PieceToId("<0x08>")
        self._last_control = self._model.PieceToId("<0x08>")

    @property
    def eos_id(self) -> int:
        """Return the end-of-sentence token ID.

        This token must be inserted to separate independent sequences. It must be the first token
        of each sequence in the batch, too.
        """
        return self._model.eos_id()

    @property
    def unk_id(self) -> int:
        """Return the unknown token ID - most likely, zero."""
        return self._model.unk_id()

    def id_of(self, token: str) -> int:
        """Return the ID of the given token."""
        return self._model.PieceToId(token)

    @property
    def vocab_size(self) -> int:
        """Return the number of unique tokens."""
        return self._model.vocab_size()

    def encode(self, input: Sequence[Chunk] | str | bytes) -> torch.Tensor:
        """Tokenize a text or a series of chunks to a Torch tensor of int32 dtype.

        If the input is a string or bytes, implicitly use the MIXED modality, i.e. tokenize
        with preserved whitespaces.

        Do not change the code so that `input` accepts `Chunk` additionally to `Sequence[Chunk]`.

        :return: A CPU tensor with the token IDs.
        """
        if isinstance(input, str | bytes):
            return self._encode_mixed_to_tensor(Chunk(contents=input, mod=Modality.MIXED))
        encoded_chunks = [self._encode_chunk(chunk) for chunk in input]
        result = torch.empty(
            sum(len(chunk) for chunk in encoded_chunks),
            dtype=torch.int32,
            device="cpu",
        )
        return torch.concatenate(encoded_chunks, out=result)

    def _encode_chunk(self, chunk: Chunk) -> torch.Tensor:
        match chunk.mod:
            case Modality.MIXED:
                return self._encode_mixed_to_tensor(chunk)
            case Modality.NATURAL:
                return self._encode_natural_to_tensor(chunk)
            case Modality.CODE:
                return self._encode_code_to_tensor(chunk)
            case Modality.CONTROL:
                return self._encode_control_to_tensor(chunk)
        raise AssertionError(f"Missed modality: {chunk.mod}")

    def _encode_mixed_to_tensor(self, chunk: Chunk) -> torch.Tensor:
        return torch.tensor(list(self.encode_mixed(chunk)), dtype=torch.int32, device="cpu")

    def _encode_natural_to_tensor(self, chunk: Chunk) -> torch.Tensor:
        return torch.tensor(list(self.encode_natural(chunk)), dtype=torch.int32, device="cpu")

    def encode_mixed(self, chunk: Chunk) -> Iterator[int]:
        """Yield the encoded text with whitespaces preserved (document serialization)."""
        assert chunk.mod == Modality.MIXED
        contents = chunk.contents.encode() if isinstance(chunk.contents, str) else chunk.contents
        yield from self._model_encode(self._verbatim * 2 + contents)

    def encode_natural(self, chunk: Chunk) -> Iterator[int]:
        """Yield the encoded natural text (document serialization)."""
        assert chunk.mod == Modality.NATURAL
        yield from self._model_encode(chunk.contents)

    _quirks_encode_str = {
        "¹": "^1",
        "²": "^2",
        "³": "^3",
    }

    _quirks_encode_bytes = {k.encode(): v.encode() for k, v in _quirks_encode_str.items()}

    _quirks_decode = {
        "帱": "^1",
        "帲": "^2",
        "帳": "^3",
    }

    def _model_encode(self, text: str | bytes) -> list[int]:
        for key, val in (
            self._quirks_encode_bytes if isinstance(text, bytes) else self._quirks_encode_str
        ).items():
            text = text.replace(key, val)
        return self._model.encode(text)

    def _model_decode(self, tokens: list[int]) -> str:
        result = self._model.decode(tokens)
        for key, val in self._quirks_decode.items():
            result = result.replace(key, val)
        return result

    def _encode_code_to_tensor(self, chunk: Chunk) -> torch.Tensor:
        result = [
            *self.encode_code_header(chunk),
            *self.encode_code_contents_start(),
            *self.encode_code_contents(chunk),
            *self.encode_code_contents_finish(),
        ]
        return torch.tensor(result, dtype=torch.int32, device="cpu")

    def encode_code_header(self, chunk: Chunk) -> Iterator[int]:
        """Yield the encoded code header (document serialization)."""
        assert chunk.mod == Modality.CODE
        if not chunk.code_file_path:
            return
        # https://github.com/poolsideai/forge/blob/main/pkg/data/serializer/serializer.go
        # https://github.com/poolsideai/sentencepiece/blob/staging/src/spm_encode_main.cc#L295
        yield self._ps_code_meta_start
        yield from self._model_encode(f"File path: {chunk.code_file_path}\n")
        yield self._ps_code_meta_end

    def encode_code_contents_start(self) -> Iterator[int]:
        """Yield the code block start token (document serialization)."""
        yield self._ps_code_start

    def encode_code_contents(self, chunk: Chunk) -> Iterator[int]:
        """Yield encoded code contents (document serialization)."""
        assert chunk.mod == Modality.CODE
        encode = self._model_encode
        # https://github.com/poolsideai/sentencepiece/blob/staging/src/spm_encode_main.cc#L230
        contents = chunk.contents.encode() if isinstance(chunk.contents, str) else chunk.contents
        if chunk.code_line_start is None:
            yield from encode(self._verbatim * 2 + contents)
        else:
            for i, line in enumerate(
                contents.splitlines(keepends=True),
                start=chunk.code_line_start,
            ):
                yield self._ps_line_delim
                if chunk.code_line_start:
                    yield from encode(str(i))
                # hack to preserve leading whitespace
                yield from encode(self._verbatim * 2 + line)

    def encode_code_contents_finish(self) -> Iterator[int]:
        """Yield the code block finish token (document serialization)."""
        yield self._ps_code_end

    def _encode_control_to_tensor(self, chunk: Chunk) -> torch.Tensor:
        return torch.tensor(list(self.encode_control(chunk)), dtype=torch.int32, device="cpu")

    def encode_control(self, chunk: Chunk) -> Iterator[int]:
        """Yield encoded control command (document serialization)."""
        assert chunk.mod == Modality.CONTROL
        yield self._ps_xcontrol_start
        yield from self._model_encode(chunk.contents)
        yield self._ps_xcontrol_end

    @torch.no_grad
    def decode_flat(self, input: torch.Tensor) -> str:
        """Convert the tensor with token IDs to corresponding text."""
        control_mask = (input <= self._last_control) & (input >= self._first_control)
        loose_input = input.detach().clone()
        loose_input[control_mask] = self.eos_id  # these tokens will be ignored
        if loose_input[0] == self._ps_code_start:
            loose_input[0] = self.eos_id

        if len(
            code_splits := torch.nonzero(
                (loose_input == self._ps_code_start)
                | (input == self._ps_line_delim)
                | (input == self._ps_xcontrol_start),
            ).squeeze(-1),
        ):
            chunks = torch.tensor_split(loose_input, code_splits)
        else:
            chunks = [loose_input]
        return "".join(self._model_decode(chunk.type(torch.int32).tolist()) for chunk in chunks)

    def decode_structured(self, input: torch.Tensor) -> list[Chunk]:
        """Inverse encode(); parse LLM output to structured representation."""
        split_mask = (
            (input == self._ps_code_meta_start)
            | (input == self._ps_code_meta_end)
            | (input == self._ps_code_start)
            | (input == self._ps_code_end)
            | (input == self._ps_xcontrol_start)
            | (input == self._ps_xcontrol_end)
        )
        splits = torch.tensor_split(input, torch.nonzero(split_mask).squeeze(-1))
        parsed = []
        decode = self._model_decode

        class BlockType:
            META = 0
            CODE = 1
            CODE_LINES = 2
            MIXED = 3
            CONTROL = 4

        for split in splits:
            if len(split) == 0:
                continue
            match split[0]:
                case self._ps_code_meta_start:
                    parsed.append((BlockType.META, decode(split[1:].to(torch.int32).tolist())))
                case self._ps_code_meta_end:
                    continue
                case self._ps_code_start:
                    split = split[1:].detach().clone()
                    line_splits = torch.nonzero(split == self._ps_line_delim).squeeze(-1)
                    split[line_splits] = self.eos_id
                    lines = torch.tensor_split(split, line_splits)
                    contents = "".join(
                        decode(
                            (
                                line[1:]
                                if line.numel() > 0 and line[0] == self._ps_line_delim
                                else line
                            )
                            .to(torch.int32)
                            .tolist(),
                        )
                        for line in lines
                    )
                    parsed.append(
                        (
                            BlockType.CODE_LINES if line_splits.numel() else BlockType.CODE,
                            contents,
                        ),
                    )
                case self._ps_xcontrol_start:
                    parsed.append((BlockType.CONTROL, decode(split[1:].to(torch.int32).tolist())))
                case _:
                    if split[0] == self._ps_code_end or split[0] == self._ps_xcontrol_end:
                        split = split[1:]
                    parsed.append((BlockType.MIXED, decode(split.to(torch.int32).tolist())))
        chunks = []
        for i, (block_type, contents) in enumerate(parsed):
            if not contents:
                continue
            match block_type:
                case BlockType.META:
                    continue
                case BlockType.CODE | BlockType.CODE_LINES:
                    if i > 0 and parsed[i - 1][0] == BlockType.META:
                        path = parsed[i - 1][1].replace("File path: ", "").rstrip("\n")
                    else:
                        path = None
                    chunks.append(
                        Chunk(
                            mod=Modality.CODE,
                            contents=contents,
                            code_file_path=path,
                            code_line_start=None if block_type == 1 else 0,
                        ),
                    )
                case BlockType.MIXED:
                    chunks.append(Chunk(mod=Modality.MIXED, contents=contents))
                case BlockType.CONTROL:
                    chunks.append(Chunk(mod=Modality.CONTROL, contents=contents))
        return chunks
