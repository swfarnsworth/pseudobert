import typing as t
import typing_extensions as te

from bratlib import data as bd
from spacy.tokens.span import Span

MASK = '[MASK]'


class PseudoSentence(te.Protocol):
    sent: str
    score: float
    pos_match: bool


SentenceGenerator = t.Generator[PseudoSentence, None, None]
SentenceFilter = t.Callable[[PseudoSentence], bool]


def default_filter(ps: PseudoSentence) -> bool:
    """Returns False for None or PseudoSentence instances for which the POS
    of the original and prediction do not match"""
    return ps is not None and ps.pos_match


def adjust_spans(ent: bd.Entity, offset: int) -> t.Tuple[int, int]:
    start = ent.spans[0][0] + offset
    end = ent.spans[-1][-1] + offset
    ent.spans = [(start, end)]
    return start, end


def find_sentence(start: int, sentences: t.List[t.Tuple[range, Span]]) -> Span:
    for r, s in sentences:
        if start in r:
            return s
    raise RuntimeError(f'start value {start} not in any range')  # This should never happen
