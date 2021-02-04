from pathlib import Path

from bratlib.data import BratDataset
from joblib import Parallel, delayed
from more_itertools import chunked

from pseudobert.pseudofiers.relations import PseudoBertRelater


@delayed
def _parallel_pseudofy(files: BratDataset, output_dir: Path) -> None:
    relater = PseudoBertRelater.init_scientific()
    relater.pseudofy_dataset(files, output_dir)


def parallel_psueodfy(dataset: BratDataset, output_dir: Path, num_jobs: int) -> None:
    directory = dataset.directory
    Parallel(n_jobs=num_jobs)(_parallel_pseudofy(BratDataset(directory, files), output_dir)
                                                 for files in chunked(dataset, num_jobs))
