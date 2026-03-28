from pathlib import Path

import torch

from fgdn_model import build_fgdn_model


def main():
    dataset_path = Path(
        r"D:/FGDN_Project/data/processed/pyg_datasets/AAL/5_fold/fold_1/train_dataset.pt"
    )

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    sample = dataset[0]

    # Simulate a batched graph with batch vector of zeros
    sample.batch = torch.zeros(sample.x.size(0), dtype=torch.long)

    model = build_fgdn_model(
    num_node_features=sample.x.size(1),
    num_nodes=sample.x.size(0),
    )
    model.eval()

    with torch.no_grad():
        logits, embedding = model(sample)

    print("Sample x shape           :", tuple(sample.x.shape))
    print("ASD edge_index shape     :", tuple(sample.edge_index_asd.shape))
    print("HC edge_index shape      :", tuple(sample.edge_index_hc.shape))
    print("Output logits shape      :", tuple(logits.shape))
    print("Graph embedding shape    :", tuple(embedding.shape))


if __name__ == "__main__":
    main()