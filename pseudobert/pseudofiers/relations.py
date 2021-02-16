import logging
import typing as t
from copy import deepcopy
from dataclasses import dataclass

from bratlib import data as bd
from bratlib.data.extensions.annotation_types import ContigEntity
from spacy.tokens.span import Span

from pseudobert.pseudofiers import _utils, base_pseudofier

MASK = _utils.MASK


@dataclass
class PseudoSentence(_utils.PseudoSentence):
    rel: bd.Relation
    sent: str
    score: float
    pos_match: bool


def _make_contig_rel(rel: bd.Relation) -> t.Optional[bd.Relation]:
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


class PseudoBertRelater(base_pseudofier.BasePseudofier):
    """
    This class contains the methods for generating pseudo relational data. The methods are provided in this
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

    def _pseudofy_side(self, rel: bd.Relation, sentence: Span, do_left=True) -> _utils.SentenceGenerator:
        rel = _make_contig_rel(rel)
        if not rel:
            return
        # _make_contig_rel does make a deep copy but no interesting changes have been made yet

        ent, other_ent = (rel.arg1, rel.arg2) if do_left else (rel.arg2, rel.arg1)
        original_token = ent.mention

        text = str(sentence)
        span_start = sentence.start_char

        start, end = _utils.adjust_spans(ent, -span_start)
        _utils.adjust_spans(other_ent, -span_start)

        try:
            original_pos = [tok.pos_ for tok in sentence.char_span(start, end)]
        except TypeError:
            # The char span doesn't line up with any tokens,
            # thus we can't figure out if the prediction is the right POS
            self._output_log.append((text, original_token, None, None, None, 'TOKENIZATION DOES NOT ALIGN'))
            return None

        for token, score in self._call_bert(text, start, end):
            new_sent = text[:start] + token + text[end:]
            new_doc = self.nlp(new_sent)
            new_span = new_doc.char_span(start, start + len(token))

            if new_span is None:
                self._output_log.append((text, original_token, token, None, None, 'NO POS MATCH'))
                continue

            pos_match = [tok.pos_ for tok in new_span] == original_pos

            this_rel = deepcopy(rel)
            ent, other_ent = (this_rel.arg1, this_rel.arg2) if do_left else (this_rel.arg2, this_rel.arg1)

            ent.spans = [(start, start + len(token))]

            if ent.start < other_ent.start:
                # If the entity being changed comes before the one not being changed, the spans of the other must
                # also be adjusted; it is not guaranteed that `rel.arg1` always comes before `rel.arg2`
                new_offset = len(token) - len(ent.mention)
                _utils.adjust_spans(other_ent, new_offset)

            ent.mention = token

            new_ps = PseudoSentence(this_rel, new_sent, float(score), pos_match)
            self._output_log.append((text, original_token, token, float(score), pos_match, None))
            yield new_ps

    def _pseudofy_instance(self, rel: bd.Relation, sentence: Span) -> _utils.SentenceGenerator:
        """
        Generates pseudo instances given a relation and the spaCy Span in which it occurs. Passing a Span that does
        not contain the relation given by `rel` has undefined behavior and will probably have incoherent results.
        """
        yield from self._pseudofy_side(rel, sentence, True)
        yield from self._pseudofy_side(rel, sentence, False)

    def pseudofy_file_generator(self, ann: bd.BratFile) -> _utils.SentenceGenerator:
        text = ann.txt_path.read_text()
        doc = self.nlp(text)
        sentence_ranges = [(range(sent.start_char, sent.end_char + 1), sent) for sent in doc.sents]

        for rel in ann.relations:
            rel = _make_contig_rel(rel)
            if not rel:
                continue

            # Identify which sentence each entity is from
            sentences = [_utils.find_sentence(arg, sentence_ranges) for arg in
                         (rel.arg1.start, rel.arg1.end, rel.arg2.start, rel.arg2.end)]

            if not (sentences[0] is sentences[1] is sentences[2] is sentences[3]):
                # For this stage of development, we are only supporting relations contained in
                # a single sentence, but we plan to progress
                continue

            text_span = sentences[0]

            yield from filter(self.filter, self._pseudofy_instance(rel, text_span))

    def pseudofy_file(self, ann: bd.BratFile) -> base_pseudofier.PseudoBratFile:
        logging.info(f'Pseudofying file: {ann.ann_path}')

        new_relations = []
        new_entities = []

        output_txt = ''
        output_offset = 0

        for pseudsent in self.pseudofy_file_generator(ann):
            output_txt += pseudsent.sent
            new_rel = pseudsent.rel

            _utils.adjust_spans(new_rel.arg1, output_offset)
            _utils.adjust_spans(new_rel.arg2, output_offset)

            new_relations.append(new_rel)
            new_entities += [new_rel.arg1, new_rel.arg2]

            output_offset += len(pseudsent.sent)

        return base_pseudofier.PseudoBratFile.from_pseudo_data(
            original_ann=ann,
            text=output_txt,
            entities=sorted(new_entities),
            relations=sorted(new_relations)
        )
