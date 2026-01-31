from dataclasses import dataclass

import hazm
from PersianG2p import Persian_g2p_converter


@dataclass
class G2PResult:
    word: str
    pronunciation: str
    error: str | None


_normalizer = None


def _get_normalizer():
    global _normalizer
    if _normalizer is None:
        _normalizer = hazm.Normalizer()
    return _normalizer


class CachedPersianG2P:
    def __init__(self, use_large=True):
        self.converter = Persian_g2p_converter(use_large=use_large)
        self.normalizer = _get_normalizer()

    def transliterate(self, word, tidy=True):
        norm_word = self.normalizer.normalize(word)
        tokens = hazm.word_tokenize(norm_word)

        result_parts = []
        for tok in tokens:
            if not any(c in tok for c in self.converter.graphemes):
                result_parts.append(tok)
            elif tok in self.converter.tihu:
                result_parts.extend([' ', self.converter.tihu[tok], ' '])
            else:
                result_parts.extend(self.converter.predict(tok))
            result_parts.append(' ')

        output = ''.join(result_parts[:-1]) if result_parts else ''

        if tidy:
            output = Persian_g2p_converter.convert_from_native_to_good(output)

        return output


def batch_transcribe(words, use_large=True, tidy=True):
    conv = CachedPersianG2P(use_large=use_large)
    results = []

    for w in words:
        word = str(w).strip()
        if not word:
            continue

        try:
            pron = conv.transliterate(word, tidy=tidy)
            results.append(G2PResult(word=word, pronunciation=pron, error=None))
        except Exception as e:
            results.append(G2PResult(word=word, pronunciation="", error=str(e)))

    return results
