import argparse
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import spacy
import torch
import transformers as tfs
from bratlib import data as brat_data
from bratlib.data.extensions.instance import ContigEntity
from spacy.tokens.span import Span

nlp = spacy.load('en_core_sci_lg')
bert = tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
bert_tokenizer = tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


@dataclass
class PseudoSentence:
    rel: brat_data.Relation
    sent: str
    score: float
    pos_match: bool


SentenceGenerator = t.Generator[PseudoSentence, None, None]
SentenceFilter = t.Callable[[PseudoSentence], bool]


def _default_filter(ps: PseudoSentence):
    return ps is not None and ps.pos_match


def _make_contig_rel(rel: brat_data.Relation) -> t.Union[brat_data.Relation, None]:
    """
    Validator that creates a deep copy of a Relation where both args are converted to
    ContigEntity, or returns None to reject Relations for which the args don't
    represent contiguous mentions.
    """
    if len(rel.arg1.spans) != 1 or len(rel.arg2.spans) != 1:
        return None
    rel = deepcopy(rel)
    rel.arg1.__class__ = ContigEntity
    rel.arg2.__class__ = ContigEntity
    return rel


def adjust_spans(ent: brat_data.Entity, offset: int) -> t.Tuple[int, int]:
    start = ent.spans[0][0] + offset
    end = ent.spans[-1][-1] + offset
    ent.spans = [(start, end)]
    return start, end


def find_sentence(start: int, sentences: t.List[t.Tuple[range, Span]]) -> Span:
    for r, s in sentences:
        if start in r:
            return s
    raise RuntimeError(f'start value {start} not in any range')


def _pseudofy_side(rel: brat_data.Relation, sentence: Span, k: int, do_left=True) -> SentenceGenerator:

    rel = _make_contig_rel(rel)
    if not rel:
        return

    ent, other_ent = (rel.arg1, rel.arg2) if do_left else (rel.arg2, rel.arg1)

    text = str(sentence)
    span_start = sentence.start_char

    start, end = adjust_spans(ent, -span_start)
    adjust_spans(other_ent, -span_start)

    try:
        original_pos = [t.pos_ for t in sentence.char_span(start, end)]
    except TypeError:
        # The char span doesn't line up with any tokens,
        # thus we can't figure out if the prediction is the right POS
        return None

    masked_sentence = text[:start] + '[MASK]' + text[end:]
    tokenized_sentence = bert_tokenizer.tokenize(masked_sentence)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)
    token_tensor = torch.tensor(indexed_tokens)
    mask_tensor = torch.tensor([token != '[MASK]' for token in tokenized_sentence], dtype=torch.float)

    with torch.no_grad():
        result = bert(token_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), masked_lm_labels=None)

    result = result[0].squeeze(0)
    scores = torch.softmax(result, dim=-1)
    mask_index = tokenized_sentence.index('[MASK]')

    topk_scores, topk_indices = torch.topk(scores[mask_index, :], k, sorted=True)
    topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices)

    for token, score in zip(topk_tokens, topk_scores):
        new_sent = text[:start] + token + text[end:]
        new_doc = nlp(new_sent)
        new_span = new_doc.char_span(start, start + len(token))

        if new_span is None:
            continue

        pos_match = [t.pos_ for t in new_span] == original_pos

        this_rel = deepcopy(rel)
        ent, other_ent = (this_rel.arg1, this_rel.arg2) if do_left else (this_rel.arg2, this_rel.arg1)

        ent.spans = [(start, start + len(token))]

        if ent.start < other_ent.start:
            new_offset = len(token) - len(ent.mention)
            other_ent.spans = [(other_ent.start + new_offset, other_ent.end + new_offset)]

        ent.mention = token

        yield PseudoSentence(this_rel, new_sent, float(score), pos_match)


def pseudofy_relation(rel: brat_data.Relation, sentence: Span, k=1) -> SentenceGenerator:
    yield from _pseudofy_side(rel, sentence, k, True)
    yield from _pseudofy_side(rel, sentence, k, False)


def pseudofy_file(ann: brat_data.BratFile) -> SentenceGenerator:
    with ann.txt_path.open() as f:
        text = f.read()
    doc = nlp(text)
    sentences = [(range(sent.start_char, sent.end_char + 1), sent) for sent in doc.sents]

    for rel in ann.relations:
        rel = _make_contig_rel(rel)
        if not rel:
            continue

        # Make sure all entities are in the same sentence
        ent1_sent_a = find_sentence(rel.arg1.start, sentences)
        ent1_sent_b = find_sentence(rel.arg1.end, sentences)
        ent2_sent_a = find_sentence(rel.arg2.start, sentences)
        ent2_sent_b = find_sentence(rel.arg2.end, sentences)

        if not (ent1_sent_a is ent1_sent_b is ent2_sent_a is ent2_sent_b):
            continue

        yield from filter(_default_filter, pseudofy_relation(rel, ent1_sent_a))


def _psudofy_file(ann: brat_data.BratFile, output_dir: Path) -> None:
    pseudo_ann = output_dir / ('pseudo_' + ann.name + '.ann')
    pseudo_txt = output_dir / ('pseudo_' + ann.name + '.txt')

    new_relations = []
    new_entities = []

    output_txt = ''
    output_offset = 0

    for pseudsent in pseudofy_file(ann):
        output_txt += pseudsent.sent
        new_rel = pseudsent.rel

        adjust_spans(new_rel.arg1, output_offset)
        adjust_spans(new_rel.arg2, output_offset)

        new_relations.append(new_rel)
        new_entities += [new_rel.arg1, new_rel.arg2]

        output_offset += len(pseudsent.sent)

    with pseudo_txt.open('w') as f:
        f.write(output_txt)

    new_ann = object.__new__(brat_data.BratFile)
    new_ann.__dict__ = {'_entities': sorted(new_entities), '_relations': sorted(new_relations)}

    with pseudo_ann.open('w+') as f:
        f.write(str(new_ann))


def pseudofy_dataset(dataset: brat_data.BratDataset, output_dir: Path) -> brat_data.BratDataset:
    for ann in dataset:
        _psudofy_file(ann, output_dir)
    return brat_data.BratDataset.from_directory(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('output_directory')
    args = parser.parse_args()

    dataset = brat_data.BratDataset.from_directory(args.input_dataset)
    output_dir = Path(args.output_directory)

    pseudofy_dataset(dataset, output_dir)


if __name__ == '__main__':
    main()
