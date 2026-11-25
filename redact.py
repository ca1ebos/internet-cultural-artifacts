import argparse
from pathlib import Path
import re

from common import read_rows, write_rows

HANDLE = re.compile(r'(?i)(?<!\w)(?:/?u/[a-z0-9_-]+|@[a-z0-9_.-]{2,})')
ENTITY_TYPES = {'PERSON': 'someone', 'GPE': 'a place', 'LOC': 'a place', 'FAC': 'a venue'}


def redact(source, infile, outfile):
    import spacy

    nlp = spacy.load('en_core_web_sm', exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])

    def items():
        for row in read_rows(infile):
            for field in ('question', 'answer'):
                yield HANDLE.sub('someone', row[field]), (row, field)

    def results():
        for doc, (row, field) in nlp.pipe(items(), as_tuples=True, batch_size=32):
            text = doc.text
            for entity in reversed(doc.ents):
                replacement = ENTITY_TYPES.get(entity.label_)
                if source == 'linkedin' and entity.label_ == 'ORG':
                    replacement = 'a company'
                if replacement:
                    text = text[:entity.start_char] + replacement + text[entity.end_char:]
            row[field] = text
            if field == 'answer':
                yield row

    write_rows(outfile, results())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['quora', 'linkedin', 'reddit'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    redact(args.source, args.input, args.output)
