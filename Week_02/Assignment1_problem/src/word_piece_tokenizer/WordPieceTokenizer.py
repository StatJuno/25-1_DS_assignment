from collections import OrderedDict
from pathlib import Path
import os

from .BasicTokenizer import BasicTokenizer
from .utils import load_vocab, whitespace_tokenize

class WordpieceTokenizer:
    """WordPiece 토크나이저 클래스입니다."""
    def __init__(self, max_input_chars_per_word=100):
        vocab_path = os.path.join(str(Path(__file__).resolve().parent), "vocab.txt")
        self.vocab = load_vocab(vocab_path)
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        텍스트를 WordPiece 토큰으로 변환합니다.
        이 함수는 주어진 어휘를 기반으로 Greedy Longest-Match-First 알고리즘을 사용하여 토큰화합니다.
        
        예시: input = "unaffable" -> 출력: ["un", "##aff", "##able"]

        Args:
            text: BasicTokenizer를 거친 단일 토큰 또는 공백으로 구분된 토큰들

        Returns:
            WordPiece 토큰들의 리스트. 시작 토큰([CLS])과 종료 토큰([SEP])이 포함됩니다.
        """
        output_tokens = []
        # 입력 텍스트를 공백 기준으로 토큰화합니다.
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 단어가 너무 길면 [UNK] 토큰을 추가합니다.
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return [self.cls_token] + output_tokens + [self.sep_token]
