import re
import typing as t
from sys import argv

from bratlib import data as bd

from pseudobert.create_rels import PseudoSentence as PS

log_file = argv[1]
pattern = re.compile(r'INFO:root:Original instance: (.*)\nINFO:root:New instance: (.*)')

with open(log_file) as f:
    log_text = f.read()

data: t.List[t.Tuple[bd.Relation, PS]] = [(exec(m[1]), exec((m[2]))) for m in pattern.finditer(log_text)]

print(f'Found {len(data)} instances')
