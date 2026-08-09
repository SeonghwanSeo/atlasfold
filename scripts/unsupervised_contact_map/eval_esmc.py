import os
import pickle
import shutil

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from atlaslm.pretrained import load_model


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return (x + x.transpose(-1, -2)) / 2


def apc(x: torch.Tensor) -> torch.Tensor:
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum((-1, -2), keepdim=True)
    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


class ESMStructuralSplitDataset(torch.utils.data.Dataset):
    """ESM Structural Split Dataset
    https://github.com/facebookresearch/esm/blob/main/esm/data.py

    For each SCOPe classification level of family/superfamily/fold (in order of difficulty),
    we have split the data into 5 partitions for cross validation. These are provided
    in a downloaded splits folder, in the format:
            splits/{split_level}/{cv_partition}/{train|valid}.txt
    where train is the partition and valid is the concatentation of the remaining 4.

    For each SCOPe domain, we provide a pkl dump that contains:
        - seq    : The domain sequence, stored as an L-length string
        - ssp    : The secondary structure labels, stored as an L-length string
        - dist   : The distance map, stored as an LxL numpy array
        - coords : The 3D coordinates, stored as an Lx3 numpy array
    """  # noqa

    base_folder = "structural-data"
    file_list = [
        #  url  tar filename   filename      MD5 Hash
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz",
            "splits.tar.gz",
            "splits",
            "456fe1c7f22c9d3d8dfe9735da52411d",
        ),
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz",
            "pkl.tar.gz",
            "pkl",
            "644ea91e56066c750cd50101d390f5db",
        ),
    ]

    def __init__(
        self,
        split_level: str,
        cv_partition: str,
        split: str,
        root_path: str = "./data/esm_structural_split",
        download=False,
    ):
        super().__init__()
        assert split in ["train", "valid"], "train_valid must be 'train' or 'valid'"
        self.root_path = root_path
        self.base_path = os.path.join(self.root_path, self.base_folder)
        self.pkl_dir = os.path.join(self.base_path, "pkl")

        # check if root path has what you need or else download it
        if download:
            self.download()

        self.split_file = os.path.join(
            self.base_path, "splits", split_level, cv_partition, f"{split}.txt"
        )
        with open(self.split_file) as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        """
        Returns a dict with the following entires
         - seq : Str (domain sequence)
         - ssp : Str (SSP labels)
         - dist : np.array (distance map)
         - coords : np.array (3D coordinates)
        """
        name = self.names[index]
        pkl_fname = os.path.join(self.pkl_dir, name[1:3], f"{name}.pkl")
        with open(pkl_fname, "rb") as f:
            obj = pickle.load(f)
        return obj

    def _check_exists(self) -> bool:
        for _, _, filename, _ in self.file_list:
            fpath = os.path.join(self.base_path, filename)
            if not os.path.exists(fpath) or not os.path.isdir(fpath):
                return False
        return True

    def download(self):
        from torchvision.datasets.utils import download_url

        if self._check_exists():
            print("Files already downloaded and verified")
            return
        for url, tar_filename, _, md5_hash in self.file_list:
            download_path = os.path.join(self.base_path, tar_filename)
            download_url(
                url=url, root=self.base_path, filename=tar_filename, md5=md5_hash
            )
            shutil.unpack_archive(download_path, self.base_path)


def download_esm_structural_split_dataset(split_level: str, cv_partition: str):
    print(
        f"Downloading dataset for "
        f"split level: {split_level}, CV partition: {cv_partition}"
    )
    ESMStructuralSplitDataset(
        split_level=split_level, cv_partition=cv_partition, split="train", download=True
    )
    ESMStructuralSplitDataset(
        split_level=split_level, cv_partition=cv_partition, split="valid", download=True
    )


def bootstrap_logistic_regression_models(
    model: ESMC,
    dataset: ESMStructuralSplitDataset,
    n_bootstraps: int = 3,
    n_samples: int = 20,
) -> list[LogisticRegression]:
    """Bootstrap sampling to fit multiple logistic regression models."""
    logreg_models: list[LogisticRegression] = []

    # Generate random indices for bootstrap samples
    random_indices: list[np.ndarray] = []
    n_train = len(dataset)
    for i in range(n_bootstraps):
        rng = np.random.default_rng(42 + i)
        random_indices.append(rng.choice(n_train, size=n_samples, replace=False))

    # Preload all data to avoid repeated disk I/O
    train_data_lists = [[dataset[i] for i in indices] for indices in random_indices]

    for trial_i in tqdm(
        range(n_bootstraps), desc="Fitting Logistic Regression Models", leave=False
    ):
        train_datas = train_data_lists[trial_i]

        # Collect features and labels
        X, Y = [], []
        for j in tqdm(
            range(len(train_datas)),
            desc=f"Processing Bootstrap Sample {trial_i + 1}",
            leave=False,
        ):
            data = train_datas[j]
            seq = data["seq"]
            L = len(seq)
            labels = data["dist"] < 8

            out = model.embed_sequences([seq], return_attentions=True)
            assert out.attentions is not None
            N, H = model.n_layers, model.n_heads
            attentions = torch.stack(out.attentions, dim=1)  # (1, N, H, L, L)

            # Filter pairs with |i-j| >= 6
            i_idx, j_idx = np.triu_indices(L, k=6)

            # Extract attention maps, removing CLS/SEP tokens
            attns = attentions[0, :, :, 1 : L + 1, 1 : L + 1]  # (N, H, L, L)
            assert attns.shape == (N, H, L, L), (
                f"Unexpected attention shape: {attns.shape}, expected "
                f"({N}, {H}, {L}, {L})"
            )

            # Symmetrize and APC
            attns = attns.cpu()
            attns = apc(symmetrize(attns))
            attns = attns.permute(2, 3, 0, 1).reshape(L, L, N * H)
            attns = attns.numpy()

            # Extract pair features
            pair_features = attns[i_idx, j_idx, :]  # (num_pairs, N*H)

            # Get labels
            pair_labels = labels[i_idx, j_idx].astype(int)  # (num_pairs,)

            X.append(pair_features)
            Y.append(pair_labels)

        X_concat = np.concatenate(X)
        Y_concat = np.concatenate(Y)

        # Fit logistic regression
        seed = 42 + trial_i
        logreg = fit_logistic_regression(X_concat, Y_concat, seed)
        logreg_models.append(logreg)

    return logreg_models


def fit_logistic_regression(
    X: np.ndarray, Y: np.ndarray, seed: int
) -> LogisticRegression:
    logreg = LogisticRegression(
        C=0.15,
        l1_ratio=1,
        solver="liblinear",
        random_state=seed,
        max_iter=50,
    )
    logreg.fit(X, Y)
    if logreg.n_iter_[0] >= logreg.max_iter:
        print(
            f"  Logistic Regression did not converge within {logreg.max_iter} iterations"
        )
    return logreg


def test(
    model: ESMC,
    split_level: str,
    cv_partition: str,
    n_bootstraps: int = 5,
    n_samples: int = 20,
):
    train_dataset = ESMStructuralSplitDataset(split_level, cv_partition, split="train")
    valid_dataset = ESMStructuralSplitDataset(split_level, cv_partition, split="valid")
    print(
        f"Split Level: {split_level}, CV Partition: {cv_partition}\n"
        f"> Train Size: {len(train_dataset)}, Valid Size: {len(valid_dataset)}"
    )

    # Fit logistic regression model on training set
    print("Fitting Logistic Regression Model on Training Set...")
    logreg_models = bootstrap_logistic_regression_models(
        model, train_dataset, n_bootstraps, n_samples
    )
    print(f"> Fitted {len(logreg_models)} logistic regression models.")

    # Get weights and bias
    params: list[tuple[torch.Tensor, torch.Tensor]] = []
    for logreg in logreg_models:
        weights = logreg.coef_.reshape(-1)  # (total_attns,)
        bias = logreg.intercept_[0]
        W = torch.tensor(weights, dtype=torch.float32, device=model.device)
        b = torch.tensor(bias, dtype=torch.float32, device=model.device)
        params.append((W, b))
    W_bootstrap = torch.stack([W for W, _ in params])  # (n_bootstraps, total_attns)
    b_bootstrap = torch.stack([b for _, b in params])  # (n_bootstraps,)

    # Validation loop
    results = [{"P@L": [], "P@L/5": []} for _ in range(len(logreg_models))]

    print("Evaluating on Validation Set...")
    chunk_size = 5
    all_datas = [valid_dataset[i] for i in range(len(valid_dataset))]
    all_datas.sort(key=lambda x: len(x["seq"]))
    for chunk_i in tqdm(
        range(0, len(all_datas), chunk_size), desc="Validation", leave=False
    ):
        chunk_datas = all_datas[chunk_i : chunk_i + chunk_size]
        sequences = [data["seq"] for data in chunk_datas]
        out = model.embed_sequences(sequences, return_attentions=True)
        assert out.attentions is not None
        N, H = model.n_layers, model.n_heads
        attentions = torch.stack(out.attentions, dim=1)  # (B, N, H, L, L)

        for data_i in range(len(chunk_datas)):
            data = chunk_datas[data_i]
            L = len(data["seq"])
            labels = data["dist"] < 8

            # Long-range contacts: |i-j| >= 24
            i_idx, j_idx = np.triu_indices(L, k=24)
            if len(i_idx) == 0:
                continue

            # Extract attention maps, removing CLS/SEP tokens
            attns = attentions[data_i, :, :, 1 : L + 1, 1 : L + 1]  # (N, H, L, L)
            assert attns.shape == (N, H, L, L), (
                f"Unexpected attention shape: {attns.shape}, expected "
                f"({N}, {H}, {L}, {L})"
            )

            # Symmetrize and APC
            attns = apc(symmetrize(attns))
            attns = attns.permute(2, 3, 0, 1).reshape(L, L, N * H)

            # GPU-accelerated logit computation
            pair_probs = (
                torch.einsum("kj,ij->ki", W_bootstrap, attns[i_idx, j_idx, :])
                + b_bootstrap[:, None]
            )  # (n_bootstraps, num_pairs)
            pair_probs = torch.sigmoid(pair_probs).cpu().numpy()
            pair_labels = labels[i_idx, j_idx].astype(int)
            for trial_i in range(n_bootstraps):
                # Sort by predicted probability
                sorted_indices = np.argsort(-pair_probs[trial_i])  # descending order
                # Compute P@L
                top_L = sorted_indices[:L]
                results[trial_i]["P@L"].append(pair_labels[top_L].mean())
                # Compute P@L/5
                top_L5 = sorted_indices[: max(1, L // 5)]
                results[trial_i]["P@L/5"].append(pair_labels[top_L5].mean())

    # Compute and print mean/std results
    p_at_l_list = np.array([np.mean(res["P@L"]).item() for res in results])
    p_at_l5_list = np.array([np.mean(res["P@L/5"]).item() for res in results])
    mean_p_at_l = np.mean(p_at_l_list)
    std_p_at_l = np.std(p_at_l_list)
    mean_p_at_l5 = np.mean(p_at_l5_list)
    std_p_at_l5 = np.std(p_at_l5_list)
    print(f"> Results over {n_bootstraps} trials:")
    print(f"  P@L: {p_at_l_list}")
    print(f"  P@L/5: {p_at_l5_list}")
    print("> Validation Results:")
    print(f"  P@L: {mean_p_at_l:.4f} ± {std_p_at_l:.4f}")
    print(f"  P@L/5: {mean_p_at_l5:.4f} ± {std_p_at_l5:.4f}")
    print()


@torch.inference_mode()
def main(debug: bool = False):
    path = "/path/to/checkpoint.ckpt"
    path = "./model_weights/v2/stage2-800k-minlr01/500k.pt"
    path = "./model_weights/v2/stage2-1m-minlr04/500k.pt"

    model_name = "esmc_3b"
    path = "./result/esmc_3b_stage1/checkpoints/checkpoint-step=000330000.ckpt"
    model_name = "esmc_300m"
    path = "pretrained"

    split_levels = ["family", "superfamily", "fold"]
    cv_partitions = ["0", "1", "2", "3", "4"]
    if debug:
        split_levels = ["superfamily"]
        cv_partitions = ["4"]

    if True:
        # Make sure to download the datasets before running this script
        print("Downloading ESM Structural Split Datasets...")
        for split_level in split_levels:
            for cv_partition in cv_partitions:
                download_esm_structural_split_dataset(split_level, cv_partition)

    if path == "pretrained":
        # Load the ESMC model (this will download the weights if not already cached)
        print(f"Loading pretrained ESM-C model : {model_name}")
        model = load_evolutionary_scale_model(model_name, "cuda").to(torch.bfloat16)
    else:
        # Load ESMC model from a specific checkpoint
        print(f"Loading ESMC model from checkpoint: {path}")
        model = load_model(path, model_name, "cuda").to(torch.bfloat16)

    # Evaluate the model on the unsupervised contact map prediction task
    print("Evaluating ESMC Model on Unsupervised Contact Map Prediction Task...")
    for split_level in split_levels:
        for cv_partition in cv_partitions:
            test(model, split_level, cv_partition, n_bootstraps=10, n_samples=20)
            # test(model, split_level, cv_partition, n_bootstraps=5, n_samples=20)


if __name__ == "__main__":
    main(debug=True)
