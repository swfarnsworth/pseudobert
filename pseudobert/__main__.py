import argparse
import logging
from pathlib import Path

from bratlib import data as brat_data

from pseudobert.pseudofiers.relations import PseudoBertRelater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset')
    parser.add_argument('output_directory')
    args = parser.parse_args()

    dataset = brat_data.BratDataset.from_directory(args.input_dataset)
    output_dir = Path(args.output_directory)

    logging.basicConfig(filename=output_dir / 'pseudobert.log', level=logging.INFO)

    pbr = PseudoBertRelater.init_scientific()
    pbr.pseudofy_dataset(dataset, output_dir)


if __name__ == '__main__':
    main()
