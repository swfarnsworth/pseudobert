import argparse
import typing as t
from copy import copy
from dataclasses import dataclass
from enum import Enum

import spacy
import torch
import transformers as tfs
from bratlib import data as brat_data
from spacy.tokens.span import Span


nlp = spacy.load('en_core_sci_lg')
bert = tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
bert_tokenizer = tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')


@dataclass
class PseudoSentence:
    rel: brat_data.Relation
    sent: str


SentenceGenerator = t.Generator[PseudoSentence, None, None]


def adjust_ent_span(ent: brat_data.Entity, offset: int) -> None:
    start, end = ent.spans[0]
    start += offset
    end += offset
    ent.spans = [(start, end)]


def find_sentence(start: int, sentences: t.Dict[range, Span]):
    for r, s in sentences.items():
        if start in r:
            return s
    raise RuntimeError(f'start value {start} not in any range')


_Side = Enum('RIGHT LEFT')

def _pseudofy_side(rel: brat_data.Relation, side: _Side):
    ent = rel.arg1 if side is _Side.RIGHT else rel.arg2



def pseudofy_relation(rel: brat_data.Relation, sentence: Span, k=3) -> SentenceGenerator:
    text = str(sentence)
    span_start = sentence.start_char

    arg1_start = rel.arg1.spans[0][0] - span_start
    arg1_end = rel.arg1.spans[0][-1] - span_start

    try:
        arg1_pos = [t.pos_ for t in sentence.char_span(arg1_start, arg1_end)]
    except:
        return None

    arg2_start = rel.arg2.spans[0][0] - span_start
    arg2_end = rel.arg2.spans[0][-1] - span_start

    arg1_masked = text[:arg1_start] + '[MASK]' + text[arg1_end:]
    arg2_masked = text[:arg2_start] + '[MASK]' + text[arg2_end:]
    # arg2_pos = [t.pos_ for t in sentence.char_span(arg2_start, arg2_end)]

    arg1_tokenized = bert_tokenizer.tokenize(arg1_masked)
    arg2_tokenized = bert_tokenizer.tokenize(arg2_masked)

    arg1_indexed_tokens = bert_tokenizer.convert_tokens_to_ids(arg1_tokenized)
    arg2_indexed_tokens = bert_tokenizer.convert_tokens_to_ids(arg2_tokenized)

    arg1_token_tensor = torch.tensor(arg1_indexed_tokens)
    arg2_token_tensor = torch.tensor(arg2_indexed_tokens)

    arg1_mask_tensor = torch.tensor([token != '[MASK]' for token in arg1_tokenized], dtype=torch.float)
    arg2_mask_tensor = torch.tensor([token != '[MASK]' for token in arg2_tokenized], dtype=torch.float)

    with torch.no_grad():
        arg1_result = bert(arg1_token_tensor.unsqueeze(0), arg1_mask_tensor.unsqueeze(0), masked_lm_labels=None)
        arg2_result = bert(arg2_token_tensor.unsqueeze(0), arg2_mask_tensor.unsqueeze(0), masked_lm_labels=None)

    arg1_result = arg1_result[0].squeeze(0)
    arg2_result = arg2_result[0].squeeze(0)

    arg1_scores = torch.softmax(arg1_result, dim=-1)
    arg2_scores = torch.softmax(arg2_result, dim=-1)

    arg1_mask_index = arg1_tokenized.index('[MASK]')
    arg2_mask_index = arg2_tokenized.index('[MASK]')

    _, topk_indices = torch.topk(arg1_scores[arg1_mask_index, :], k, sorted=True)
    arg1_topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices)

    _, topk_indices = torch.topk(arg2_scores[arg2_mask_index, :], k, sorted=True)
    arg2_topk_tokens = bert_tokenizer.convert_ids_to_tokens(topk_indices)

    for token in arg1_topk_tokens:
        new_sent = text[:arg1_start] + token + text[arg1_end:]
        new_doc = nlp(new_sent)
        new_span = new_doc.char_span(arg1_start, arg1_start + len(token))
        if new_span is None:
            print('none reject')
            continue
        if [t.pos_ for t in new_span] != arg1_pos:
            print('reject')
            continue

        print('accept!')

        new_arg1 = copy(rel.arg1)
        new_arg1.spans = [(arg1_start, arg1_start + len(token))]
        new_arg1.mention = token

        new_offset = len(token) - len(rel.arg1.mention)

        new_arg2 = copy(rel.arg2)
        new_arg2.spans = [(arg2_start + new_offset, arg2_end + new_offset)]

        new_rel = brat_data.Relation(rel.relation, new_arg1, new_arg2)
        yield PseudoSentence(new_rel, new_sent)

    # for token in arg2_topk_tokens:
    #     new_sent = text[:arg2_start] + token + text[arg2_end:]
    #     new_arg2 = copy(rel.arg2)
    #     new_arg2.spans = [(arg1_start, arg1_start + len(token))]
    #     new_arg2.mention = token
    #
    #     new_offset = len(token) - len(rel.arg2.mention)
    #
    #     new_arg1 = copy(rel.arg1)
    #     new_arg1.spans = [(arg2_start + new_offset, arg2_end + new_offset)]
    #
    #     new_rel = brat_data.Relation(rel.relation, new_arg1, new_arg2)
    #     yield PseudoSentence(new_rel, new_sent)


def pseudofy_file(ann: brat_data.BratFile) -> SentenceGenerator:
    with ann.txt_path.open() as f:
        doc = nlp(f.read())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('output_ann_name')
    parser.add_argument('output_txt_name')
    args = parser.parse_args()

    # dataset = brat_data.BratDataset.from_directory('/home/steele/datasets/n2c2/training_n2c2')
    dataset = brat_data.BratDataset.from_directory(args.input_dataset)

    output_txt = ""
    output_offset = 0

    new_relations = []
    new_entities = []

    for f in dataset:
        for pseudsent in pseudofy_file(f):
            output_txt += pseudsent.sent
            new_rel = pseudsent.rel
            new_rel.arg1.spans = [(new_rel.arg1.spans[0][0] + output_offset, new_rel.arg1.spans[0][1] + output_offset)]
            new_rel.arg2.spans = [(new_rel.arg2.spans[0][0] + output_offset, new_rel.arg2.spans[0][1] + output_offset)]
            output_offset += len(pseudsent.sent)
            new_relations.append(new_rel)
            new_entities += [new_rel.arg1, new_rel.arg2]

    new_ann = type('Temp', (object,), {})()
    new_ann.__dict__ = {'entities': sorted(new_entities), 'relations': sorted(new_relations)}

    ann_doc = brat_data.BratFile.__str__(new_ann)

    with open(args.output_ann_name, 'w+') as f:
        f.write(ann_doc)

    with open(args.output_txt_name, 'w+') as f:
        f.write(output_txt)


if __name__ == '__main__':
    main()
