from datetime import datetime

import torch
from ase.io import read
from tqdm import tqdm

from eninet.data.data_config import DEFAULT_FLOATDTYPE
from eninet.graph.converter import Molecule2Graph
from eninet.model.config import load_config
from eninet.model.pl_wrapper import ScalarPredModule
from eninet.script.train_utils import parse, print_logo, read_dataset_from_json

torch.set_default_dtype(DEFAULT_FLOATDTYPE)


def main():
    print_logo()

    args = parse()
    config = load_config(args.config)
    print(config)

    assert args.ckpt is not None, "Please provide a checkpoint path for inference"
    ckpt_path = args.ckpt

    assert args.infer is not None, "Please provide a dataset path for inference"
    infer_path = args.infer

    if infer_path.endswith(".json"):
        mol_ids, infer_struct = read_dataset_from_json(infer_path, infer_mode=True)
    elif infer_path.endswith(".xyz"):
        infer_struct = read(infer_path, index=":")
        mol_ids = [struct.get_chemical_symbols() for struct in infer_struct]
    else:
        raise ValueError(
            "Unsupported file format for inference. Please provide a .json or .xyz file."
        )

    # Setup Converter
    converter = Molecule2Graph(cutoff=config.data.cutoff)

    model = ScalarPredModule.load_from_checkpoint(ckpt_path)

    results_file = None
    if len(infer_struct) > 1:
        results_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(results_file, "w") as f:
            f.write("mol_id,pred\n")

    for mol_id, struct in tqdm(zip(mol_ids, infer_struct), total=len(mol_ids)):
        graph = converter.build_graph(struct).to(model.device)
        linegraph = (
            converter.build_line_graph(graph).to(model.device)
            if config.model.use_linegraph
            else None
        )
        prediction = model(graph, linegraph)

        if results_file:
            with open(results_file, "a") as f:
                f.write(f"{mol_id},{prediction.item()}\n")
        else:
            print("\n" + "=" * 50)
            print(f"Prediction for structure {infer_path}: {prediction.item()}")
            print("=" * 50 + "\n")

    if results_file:
        print("\n" + "=" * 50)
        print(f"Prediction for molecules saved to {results_file}")
        print("=" * 50 + "\n")

    elif len(infer_struct) == 0:
        raise ValueError("No structures found in the dataset")


if __name__ == "__main__":
    main()
