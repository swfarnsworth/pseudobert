import abc
import typing as t
from pathlib import Path

import pandas as pd
import spacy
import torch
import transformers as tfs
from bratlib import data as bd
from spacy.language import Language
from spacy.tokens import Span

from . import _utils


class PseudoBratFile(bd.BratFile):

    @classmethod
    def from_pseudo_data(cls,
                         original_ann: bd.BratFile,
                         text: str,
                         entities: t.Optional[t.List[bd.Entity]] = None,
                         events: t.Optional[t.List[bd.Event]] = None,
                         relations: t.Optional[t.List[bd.Relation]] = None,
                         equivalences: t.Optional[t.List[bd.Equivalence]] = None,
                         attributes: t.Optional[t.List[bd.Attribute]] = None,
                         normalizations: t.Optional[t.List[bd.Normalization]] = None,
                         ):
        new = super().from_data(entities, events, relations, equivalences, attributes, normalizations)
        new.__class__ = cls
        new.text = text
        new.name = original_ann.name
        return new

    def to_disk(self, directory: Path) -> bd.BratFile:
        ann_path = directory / f'pseudo_{self.name}.ann'
        txt_path = directory / f'pseudo_{self.name}.txt'
        ann_path.write_text(str(self))
        txt_path.write_text(self.text)
        return bd.BratFile(ann_path, txt_path)


class BasePseudofier(abc.ABC):

    def __init__(self,
                 bert: tfs.BertForMaskedLM,
                 bert_tokenizer: tfs.BertTokenizer,
                 spacy_model: Language,
                 filter_: _utils.SentenceFilter,
                 k=1):
        self.bert = bert
        self.bert_tokenizer = bert_tokenizer
        self.nlp = spacy_model
        self.filter = filter_
        self.k = k

        # t.Tuple[sentence, original_token, new_token, probability, pos_match, note]
        self._output_log: t.List[t.Tuple[str, str, str, float, bool, str]] = []

    @classmethod
    def init_scientific(cls, filter_=_utils.default_filter, k=1):
        return cls(
            bert=tfs.BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased'),
            bert_tokenizer=tfs.BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased'),
            spacy_model=spacy.load('en_core_sci_lg'),
            filter_=filter_,
            k=k
        )

    def _call_bert(self, text: str, start: int, end: int) -> t.Generator[t.Tuple[str, float], None, None]:
        masked_sentence = text[:start] + _utils.MASK + text[end:]
        tokenized_sentence = self.bert_tokenizer.tokenize(masked_sentence)
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_sentence)
        token_tensor = torch.tensor(indexed_tokens)
        mask_tensor = torch.tensor([token != _utils.MASK for token in tokenized_sentence], dtype=torch.float)

        if len(token_tensor) > 512:
            # This is the token limit we report on, but the limit depends on the BERT model
            return None

        with torch.no_grad():
            result = self.bert(token_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), labels=None)

        result = result[0].squeeze(0)
        scores = torch.softmax(result, dim=-1)
        mask_index = tokenized_sentence.index(_utils.MASK)

        topk_scores, topk_indices = torch.topk(scores[mask_index, :], self.k, sorted=True)
        topk_tokens = self.bert_tokenizer.convert_ids_to_tokens(topk_indices)
        yield from zip(topk_tokens, topk_scores)

    def log_to_dataframe(self):
        df = pd.DataFrame(
            self._output_log,
            columns=['sentence', 'original_token', 'new_token', 'probability', 'pos_match', 'note']
        )
        self._output_log.clear()
        return df

    @abc.abstractmethod
    def _pseudofy_instance(self, annotation: bd.AnnData, sentence: Span) -> _utils.SentenceGenerator:
        pass

    @abc.abstractmethod
    def pseudofy_file(self, ann: bd.BratFile) -> PseudoBratFile:
        pass

    def pseudofy_dataset(self, dataset: bd.BratDataset, output_dir: Path) -> bd.BratDataset:
        for ann in dataset:
            self.pseudofy_file(ann).to_disk(output_dir)
        return bd.BratDataset.from_directory(output_dir)
