import re
import typing as t
from sys import argv

from bratlib.data import Relation
from bratlib.data.extensions.instance import ContigEntity

from pseudobert.create_rels import PseudoSentence

log_file = argv[1]
pattern = re.compile(r'INFO:root:Original instance: (.*)\nINFO:root:New instance: (.*)\n')

with open(log_file) as f:
    log_text = f.read()

data: t.List[t.Tuple[Relation, PseudoSentence]] = [(eval(m[1]), eval((m[2]))) for m in pattern.finditer(log_text)]

print(f'Found {len(data)} instances')
