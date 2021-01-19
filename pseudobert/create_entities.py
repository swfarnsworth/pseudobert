import logging
from dataclasses import dataclass
from pathlib import Path

import spacy
import torch
import transformers as tfs
from bratlib import data as bd
from spacy.tokens.span import Span

from . import _utils

MASK = _utils.MASK


@dataclass
class PseudoSentence(_utils.PseudoSentence):
    ent: bd.Entity
    sent: str
    score: float
    pos_match: bool


class PseudoBertEntityCreator:
    """
    This class contains the methods for generating pseudo NER data. The methods are provided in this
    class instead of as stand-alone functions so that the BERT model, spaCy model, and filter need only
    be specified once. If only the `pseudofy_dataset` method is wanted, one can create an instance but only
    keep a reference to that method.

    pseudofy_dataset = PseudoBertRelater.init_scientific().pseudofy_dataset

    :ivar bert: transformers.BertForMaskedLM
    :ivar bert_tokenizer: transformers.BertTokenizer
    :ivar spacy_model: used for the sentencizer and POS tagger
    :ivar filter_: used to accept or reject instances outputted by this class based on data contained in the
    PseudoSentence instance
    :ivar k: top k predictions to use from BERT, defaults to 1
    """

    def __init__(self, bert: tfs.BertForMaskedLM, bert_tokenizer: tfs.BertTokenizer, spacy_model,
                 filter_: _utils.SentenceFilter, k=1):
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer
        self.nlp = spacy_model
        self.filter = filter_
        self.k = k

    @classmethod
    def init_scientific(cls, filter_=_utils.default_filter):
        return cls(
            bert=tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased'),
            bert_tokenizer=tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'),
            spacy_model=spacy.load('en_core_sci_lg'),
            filter_=filter_
        )

    def pseudofy_entity(self, ent: bd.Entity, sentence: Span) -> _utils.SentenceGenerator:
        logging.info(f'Original instance: {ent}')

        text = str(sentence)
        span_start = sentence.start_char

        start, end = _utils.adjust_spans(ent, -span_start)

        try:
            original_pos = [tok.pos_ for tok in sentence.char_span(start, end)]
        except TypeError:
            # The char span doesn't line up with any tokens,
            # thus we can't figure out if the prediction is the right POS
            logging.info('Instance rejected; the spans given do not align with tokens according to the spaCy model')
            return None

        masked_sentence = text[:start] + MASK + text[end:]
        tokenized_sentence = self.bert_tokenizer.tokenize(masked_sentence)
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)
        token_tensor = torch.tensor(indexed_tokens)
        mask_tensor = torch.tensor([token != MASK for token in tokenized_sentence], dtype=torch.float)

        if len(token_tensor) > 512:
            # This is the token limit we report on, but the limit depends on the BERT model
            return None

        with torch.no_grad():
            result = self.bert(token_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), labels=None)

        result = result[0].squeeze(0)
        scores = torch.softmax(result, dim=-1)
        mask_index = tokenized_sentence.index(MASK)

        topk_scores, topk_indices = torch.topk(scores[mask_index, :], self.k, sorted=True)
        topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(topk_indices)

        for token, score in zip(topk_tokens, topk_scores):
            new_sent = text[:start] + token + text[end:]
            new_doc = self.nlp(new_sent)
            new_span = new_doc.char_span(start, start + len(token))

            if new_span is None:
                continue

            pos_match = [tok.pos_ for tok in new_span] == original_pos

            new_entity = bd.Entity(
                tag=ent.tag,
                spans=[(start, start + len(token))],
                mention=token
            )

            new_ps = PseudoSentence(new_entity, new_sent, float(score), pos_match)
            logging.info(f'New instance: {new_ps}')
            yield new_ps

    def pseudofy_file_generator(self, ann: bd.BratFile) -> _utils.SentenceGenerator:
        text = ann.txt_path.read_text()
        doc = self.nlp(text)
        sentence_ranges = [(range(sent.start_char, sent.end_char + 1), sent) for sent in doc.sents]

        for ent in ann.entities:
            if len(ent.spans) != 1:
                continue

            # Identify which sentence the entity is from
            sentences = [_utils.find_sentence(arg, sentence_ranges) for arg in [ent.spans[0][0], ent.spans[0][-1]]]

            if sentences[0] is not sentences[1]:
                # For this stage of development, we are only supporting relations contained in
                # a single sentence, but we plan to progress
                continue

            text_span = sentences[0]

            yield from filter(self.filter, self.pseudofy_entity(ent, text_span))

    def pseudofy_file(self, ann: bd.BratFile, output_dir: Path) -> bd.BratFile:
        logging.info(f'Pseudofying file: {ann.ann_path.name}')
        pseudo_ann = output_dir / ('pseudo_' + ann.name + '.ann')
        pseudo_txt = output_dir / ('pseudo_' + ann.name + '.txt')

        new_entities = []
        output_txt = ''
        output_offset = 0

        for pseudsent in self.pseudofy_file_generator(ann):
            output_txt += pseudsent.sent
            new_ent = pseudsent.ent
            _utils.adjust_spans(new_ent, output_offset)

            new_entities.append(new_ent)
            output_offset += len(pseudsent.sent)

        new_ann = bd.BratFile.from_data(entities=sorted(new_entities))

        pseudo_txt.write_text(output_txt)
        pseudo_ann.write_text(str(new_ann))

        new_ann.ann_path, new_ann._txt_path = pseudo_ann, pseudo_txt
        return new_ann

    def pseudofy_dataset(self, dataset: bd.BratDataset, output_dir: Path) -> bd.BratDataset:
        for ann in dataset:
            self.pseudofy_file(ann, output_dir)
        return bd.BratDataset.from_directory(output_dir)
