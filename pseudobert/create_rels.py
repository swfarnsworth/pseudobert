import logging
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


@dataclass
class PseudoSentence:
    rel: brat_data.Relation
    sent: str
    score: float
    pos_match: bool


SentenceGenerator = t.Generator[PseudoSentence, None, None]
SentenceFilter = t.Callable[[PseudoSentence], bool]


def _default_filter(ps: PseudoSentence):
    """Returns False for None or PseudoSentence instances for which the POS
    of the original and prediction do not match"""
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
    raise RuntimeError(f'start value {start} not in any range')  # This should never happen


class PseudoBertRelater:
    """
    This class contains the methods for generating pseudo relational data. The methods are provided in this
    class instead of as stand-alone functions so that the BERT model, spaCy model, and filter need only
    be specified once. If only the `pseudofy_dataset` method is wanted, one can create an instance but only
    keep a reference to that method.

    pseudofy_dataset = PseudoBertRelater.init_scientific().pseudofy_dataset

    :ivar bert: transformers.BertForMaskedLM
    :ivar bert_tokenizer: transformers.BertTokenizer
    :ivar spacy_model: used for the sentencizer and POS tagger
    :ivar filter_: used to accept or reject instances outputted by this class based on data contained in the PseudoSentence instance
    """

    def __init__(self, bert: tfs.BertForMaskedLM, bert_tokenizer: tfs.BertTokenizer, spacy_model, filter_: SentenceFilter):
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer
        self.nlp = spacy_model
        self.filter = filter_

    @classmethod
    def init_scientific(cls, filter_=_default_filter):
        return cls(
            bert=tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased'),
            bert_tokenizer=tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'),
            spacy_model=spacy.load('en_core_sci_lg'),
            filter_=filter_
        )

    def _pseudofy_side(self, rel: brat_data.Relation, sentence: Span, k: int, do_left=True) -> SentenceGenerator:

        rel = _make_contig_rel(rel)
        if not rel:
            return
        # _make_contig_rel does make a deep copy but no interesting changes have been made yet
        logging.info(f'Original instance: {rel}')

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
            logging.info('Instance rejected; the spans given do not align with tokens according to the spaCy model')
            return None

        masked_sentence = text[:start] + '[MASK]' + text[end:]
        tokenized_sentence = self.bert_tokenizer.tokenize(masked_sentence)
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)
        token_tensor = torch.tensor(indexed_tokens)
        mask_tensor = torch.tensor([token != '[MASK]' for token in tokenized_sentence], dtype=torch.float)

        if len(token_tensor) > 512:
            # This is the token limit we report on, but the limit depends on the BERT model
            return None

        with torch.no_grad():
            result = self.bert(token_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), masked_lm_labels=None)

        result = result[0].squeeze(0)
        scores = torch.softmax(result, dim=-1)
        mask_index = tokenized_sentence.index('[MASK]')

        topk_scores, topk_indices = torch.topk(scores[mask_index, :], k, sorted=True)
        topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(topk_indices)

        for token, score in zip(topk_tokens, topk_scores):
            new_sent = text[:start] + token + text[end:]
            new_doc = self.nlp(new_sent)
            new_span = new_doc.char_span(start, start + len(token))

            if new_span is None:
                continue

            pos_match = [t.pos_ for t in new_span] == original_pos

            this_rel = deepcopy(rel)
            ent, other_ent = (this_rel.arg1, this_rel.arg2) if do_left else (this_rel.arg2, this_rel.arg1)

            ent.spans = [(start, start + len(token))]

            if ent.start < other_ent.start:
                # If the entity being changed comes before the one not being changed, the spans of the other must
                # also be adjusted; it is not guaranteed that `rel.arg1` always comes before `rel.arg2`
                new_offset = len(token) - len(ent.mention)
                adjust_spans(other_ent, new_offset)

            ent.mention = token

            new_ps = PseudoSentence(this_rel, new_sent, float(score), pos_match)
            logging.info(f'New instance: {new_ps}')
            yield new_ps

    def pseudofy_relation(self, rel: brat_data.Relation, sentence: Span, k=1) -> SentenceGenerator:
        yield from self._pseudofy_side(rel, sentence, k, True)
        yield from self._pseudofy_side(rel, sentence, k, False)

    def pseudofy_file(self, ann: brat_data.BratFile) -> SentenceGenerator:
        with ann.txt_path.open() as f:
            text = f.read()
        doc = self.nlp(text)
        sentence_ranges = [(range(sent.start_char, sent.end_char + 1), sent) for sent in doc.sents]

        for rel in ann.relations:
            rel = _make_contig_rel(rel)
            if not rel:
                continue

            # Identify which sentence each entity is from
            sentences = [find_sentence(arg, sentence_ranges) for arg in
                         (rel.arg1.start, rel.arg1.end, rel.arg2.start, rel.arg2.end)]

            if not (sentences[0] is sentences[1] is sentences[2] is sentences[3]):
                # For this stage of development, we are only supporting relations contained in
                # a single sentence, but we plan to progress
                continue

            text_span = sentences[0]

            yield from filter(self.filter, self.pseudofy_relation(rel, text_span))

    def _pseudofy_file(self, ann: brat_data.BratFile, output_dir: Path) -> None:
        logging.info(f'Pseudofying file: {ann.ann_path}')
        pseudo_ann = output_dir / ('pseudo_' + ann.name + '.ann')
        pseudo_txt = output_dir / ('pseudo_' + ann.name + '.txt')

        new_relations = []
        new_entities = []

        output_txt = ''
        output_offset = 0

        for pseudsent in self.pseudofy_file(ann):
            output_txt += pseudsent.sent
            new_rel = pseudsent.rel

            adjust_spans(new_rel.arg1, output_offset)
            adjust_spans(new_rel.arg2, output_offset)

            new_relations.append(new_rel)
            new_entities += [new_rel.arg1, new_rel.arg2]

            output_offset += len(pseudsent.sent)

        with pseudo_txt.open('w') as f:
            f.write(output_txt)

        new_ann = brat_data.BratFile.from_data()
        new_ann._entities, new_ann._relations = sorted(new_entities), sorted(new_relations)

        with pseudo_ann.open('w+') as f:
            f.write(str(new_ann))

    def pseudofy_dataset(self, dataset: brat_data.BratDataset, output_dir: Path) -> brat_data.BratDataset:
        for ann in dataset:
            self._pseudofy_file(ann, output_dir)
        return brat_data.BratDataset.from_directory(output_dir)
