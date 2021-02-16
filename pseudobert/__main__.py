import argparse
import logging
from pathlib import Path

import joblib
import more_itertools
from bratlib import data as bd

from pseudobert.pseudofiers import _utils
from pseudobert.pseudofiers.entities import PseudoBertEntityCreator
from pseudobert.pseudofiers.relations import PseudoBertRelater
from pseudobert.pseudofiers.relations_context import PseudoBertContextRelater


pseudofiers_by_name = {
    'relations': PseudoBertRelater,
    'entities': PseudoBertEntityCreator,
    'relations_context': PseudoBertContextRelater
}


@joblib.delayed
def _parallel_pseudofy(dataset, output_dir, func):
    func(dataset, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset', help='The directory containing the hand-annotated files to augment.')
    parser.add_argument('output_directory', help='An existing directory in which to write the pseudo data.')
    parser.add_argument('action', choices=pseudofiers_by_name.keys(), help='What type of augmentation to perform.')
    parser.add_argument('-k', type=int, default=1, help='The number of new instances to attempt per hand-annotated instance.')
    parser.add_argument('-p', '--probability', type=float, default=0.0, help='The probability score a prediction needs to clear (>=) to be permitted; must be in [0, 1].')
    parser.add_argument('--num_processes', type=int, default=1, help='The number of files to process in parallel. Defaults to 1, in which case the program will run serially.')

    args = parser.parse_args()

    dataset = bd.BratDataset.from_directory(args.input_dataset)
    output_dir = Path(args.output_directory)

    run_parallel = args.num_processes > 1

    logging.basicConfig(filename=output_dir / 'pseudobert.log', level=logging.INFO)

    psb = pseudofiers_by_name[args.action].init_scientific()
    psb.k, psb.filter = args.k, _utils.Filter(args.probability)

    if not run_parallel:
        return psb.pseudofy_dataset(dataset, output_dir)

    joblib.Parallel(n_jobs=args.num_proceses)(_parallel_pseudofy(
        bd.BratDataset(dataset.directory, files),
        output_dir)
        for files in more_itertools.chunked(dataset, args.num_proceses)
    )


if __name__ == '__main__':
    main()
