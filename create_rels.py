import argparse
import typing as t
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import spacy
import torch
import transformers as tfs
from bratlib import data as brat_data
from spacy.tokens.span import Span

nlp = spacy.load('en_core_sci_lg')
bert = tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
bert_tokenizer = tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

PseudoSentence = namedtuple('PseudoSentence', 'rel sent')
SentenceGenerator = t.Generator[PseudoSentence, None, None]

def adjust_spans(ent: brat_data.Entity, offset: int) -> t.Tuple[int, int]:
    start = ent.spans[0][0] + offset
    end = ent.spans[-1][-1] + offset
    ent.spans = [(start, end)]
    return start, end


def find_sentence(start: int, sentences: t.Dict[range, Span]):
    for r, s in sentences.items():
        if start in r:
            return s
    raise RuntimeError(f'start value {start} not in any range')


def _pseudofy_side(rel: brat_data.Relation, sentence: Span, k: int, do_left=True) -> SentenceGenerator:

    if len(rel.arg1.spans) != 1 or len(rel.arg2.spans) != 1:
        return

    rel = deepcopy(rel)
    if do_left:
        ent, other_ent = rel.arg1, rel.arg2
    else:
        other_ent, ent = rel.arg1, rel.arg2

    text = str(sentence)
    span_start = sentence.start_char

    start, end = adjust_spans(ent, -span_start)
    adjust_spans(other_ent, -span_start)

    try:
        arg1_pos = [t.pos_ for t in sentence.char_span(start, end)]
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

    _, topk_indices = torch.topk(scores[mask_index, :], k, sorted=True)
    topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices)

    for token in topk_tokens:
        new_sent = text[:start] + token + text[end:]
        new_doc = nlp(new_sent)
        new_span = new_doc.char_span(start, start + len(token))

        if new_span is None:
            continue
        if [t.pos_ for t in new_span] != arg1_pos:
            continue

        rel = deepcopy(rel)
        if do_left:
            rel.arg1.spans = [(start, start + len(token))]
            new_offset = len(token) - len(ent.mention)
            rel.arg1.mention = token
            rel.arg2.spans = [(rel.arg2.spans[0][0] + new_offset, rel.arg2.spans[-1][-1] + new_offset)]
        else:
            rel.arg2.mention = token
            rel.arg2.spans = [(start, start + len(token))]

        yield PseudoSentence(rel, new_sent)


def pseudofy_relation(rel: brat_data.Relation, sentence: Span, k=3) -> SentenceGenerator:
    yield from _pseudofy_side(rel, sentence, k, True)
    yield from _pseudofy_side(rel, sentence, k, False)


def pseudofy_file(ann: brat_data.BratFile) -> SentenceGenerator:
    with ann.txt_path.open() as f:
        text = f.read()
    doc = nlp(text)
    sentences = {range(sent.start_char, sent.end_char + 1): sent for sent in doc.sents}

    for rel in ann.relations:
        # Make sure all entities are in the same sentence
        first_ent_start, first_ent_end = rel.arg1.spans[0][0], rel.arg1.spans[-1][-1]
        last_ent_start, last_ent_end = rel.arg2.spans[0][0], rel.arg2.spans[-1][-1]

        ent1_sent_a = find_sentence(first_ent_start, sentences)
        ent1_sent_b = find_sentence(first_ent_end, sentences)
        ent2_sent_a = find_sentence(last_ent_start, sentences)
        ent2_sent_b = find_sentence(last_ent_end, sentences)

        if not (ent1_sent_a is ent1_sent_b is ent2_sent_a is ent2_sent_b):
            continue

        yield from filter(None, pseudofy_relation(rel, ent1_sent_a))


def pseuofy_dataset(dataset: brat_data.BratDataset, output_dir: Path) -> brat_data.BratDataset:
    for bf in dataset:
        pseudo_ann = output_dir / ('pseudo_' + bf.name + '.ann')
        pseudo_txt = output_dir / ('pseudo_' + bf.name + '.txt')

        new_relations = []
        new_entities = []

        output_txt = pseudo_txt.open('w+')
        output_offset = 0

        for i, pseudsent in enumerate(pseudofy_file(bf)):
            if i % 100 == 0:
                output_txt.flush()
            output_txt.write(pseudsent.sent)
            new_rel = pseudsent.rel
            new_rel.arg1.spans = [(new_rel.arg1.spans[0][0] + output_offset, new_rel.arg1.spans[0][1] + output_offset)]
            new_rel.arg2.spans = [(new_rel.arg2.spans[0][0] + output_offset, new_rel.arg2.spans[0][1] + output_offset)]

            new_relations.append(new_rel)
            new_entities += [new_rel.arg1, new_rel.arg2]

            output_offset += len(pseudsent.sent)

        output_txt.close()

        new_ann = object.__new__(brat_data.BratFile)
        new_ann.__dict__ = {'_entities': sorted(new_entities), '_relations': sorted(new_relations)}

        with pseudo_ann.open('w+') as f:
            f.write(str(new_ann))

    return brat_data.BratDataset.from_directory(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('output_directory')
    args = parser.parse_args()

    dataset = brat_data.BratDataset.from_directory(args.input_dataset)
    output_dir = Path(args.output_directory)

    pseuofy_dataset(dataset, output_dir)


if __name__ == '__main__':
    main()
