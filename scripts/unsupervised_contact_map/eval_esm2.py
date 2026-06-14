import os
import pickle
import shutil

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    """(N, H, L, L) -> (N, H, L, L)"""
    return (x + x.transpose(-1, -2)) / 2


def apc(x: torch.Tensor) -> torch.Tensor:
    """(N, H, L, L) -> (N, H, L, L)"""
    a1 = x.sum(-1, keepdim=True)
    a2 = x.sum(-2, keepdim=True)
    a12 = x.sum((-1, -2), keepdim=True)
    return x - (a1 * a2) / a12


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
    model,
    alphabet,
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

    batch_converter = alphabet.get_batch_converter()
    for trial_i in tqdm(
        range(n_bootstraps), desc="Fitting Logistic Regression Models", leave=False
    ):
        train_datas = train_data_lists[trial_i]

        N, H = model.num_layers, model.attention_heads
        # Collect features and labels
        X, Y = [], []
        for i in tqdm(
            range(len(train_datas)),
            desc=f"Processing Bootstrap Sample {trial_i + 1}",
            leave=False,
        ):
            data = train_datas[i]
            seq = data["seq"]

            sequences = [(i, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
            esm_out = model.forward(batch_tokens.cuda(), need_head_weights=True)
            attentions = esm_out["attentions"]

            L = len(data["seq"])
            labels = data["dist"] < 8

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
            attns = attns.permute(2, 3, 0, 1).reshape(L, L, N * H)  # (L, L, N*H)
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
    model,
    alphabet,
    split_level: str,
    cv_partition: str,
    n_bootstraps: int = 5,
    n_samples: int = 20,
):
    batch_converter = alphabet.get_batch_converter()

    train_dataset = ESMStructuralSplitDataset(
        split_level=split_level,
        cv_partition=cv_partition,
        split="train",
        download=False,
    )
    valid_dataset = ESMStructuralSplitDataset(
        split_level=split_level,
        cv_partition=cv_partition,
        split="valid",
        download=False,
    )
    print(
        f"Split Level: {split_level}, CV Partition: {cv_partition}\n"
        f"> Train Size: {len(train_dataset)}, Valid Size: {len(valid_dataset)}"
    )

    N, H = model.num_layers, model.attention_heads

    # Fit logistic regression model on training set
    print("Fitting Logistic Regression Model on Training Set...")
    logreg_models = bootstrap_logistic_regression_models(
        model,
        alphabet,
        train_dataset,
        n_bootstraps,
        n_samples,
    )
    print(f"> Fitted {len(logreg_models)} logistic regression models.")

    # Get weights and bias
    params: list[tuple[torch.Tensor, torch.Tensor]] = []
    for logreg in logreg_models:
        weights = logreg.coef_.reshape(-1)  # (N*H,)
        bias = logreg.intercept_[0]
        W = torch.tensor(weights, dtype=torch.float32, device="cuda")
        b = torch.tensor(bias, dtype=torch.float32, device="cuda")
        params.append((W, b))
    W_bootstrap = torch.stack([W for W, _ in params])  # (n_bootstraps, N*H)
    b_bootstrap = torch.stack([b for _, b in params])  # (n_bootstraps,)

    # Validation loop
    results = [{"P@L": [], "P@L/5": []} for _ in range(len(logreg_models))]

    print("Evaluating on Validation Set...")
    chunk_size = 20
    all_datas = [valid_dataset[i] for i in range(len(valid_dataset))]
    all_datas.sort(key=lambda x: len(x["seq"]))
    for i in tqdm(range(0, len(all_datas), chunk_size), leave=False):
        chunk_datas = all_datas[i : i + chunk_size]

        sequences = [(j, data["seq"]) for j, data in enumerate(chunk_datas)]
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

        esm_out = model.forward(batch_tokens.cuda(), need_head_weights=True)
        attentions = esm_out["attentions"]

        for j in range(len(chunk_datas)):
            data = chunk_datas[j]
            L = len(data["seq"])
            labels = data["dist"] < 8

            # Long-range contacts: |i-j| >= 24
            i_idx, j_idx = np.triu_indices(L, k=24)
            if len(i_idx) == 0:
                continue

            # Extract attention maps, removing CLS/SEP tokens
            attns = attentions[j, :, :, 1 : L + 1, 1 : L + 1]
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
            for k in range(n_bootstraps):
                # Sort by predicted probability
                sorted_indices = np.argsort(-pair_probs[k])  # descending order

                # Compute P@L
                top_L = sorted_indices[:L]
                results[k]["P@L"].append(pair_labels[top_L].mean())

                # Compute P@L/5
                top_L5 = sorted_indices[: max(1, L // 5)]
                results[k]["P@L/5"].append(pair_labels[top_L5].mean())

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
    import esm

    split_levels = ["family", "superfamily", "fold"]
    cv_partitions = ["0", "1", "2", "3", "4"]
    if debug:
        split_levels = ["superfamily"]
        cv_partitions = ["4"]

    # Make sure to download the datasets before running this script
    print("Downloading ESM Structural Split Datasets...")
    for split_level in split_levels:
        for cv_partition in cv_partitions:
            download_esm_structural_split_dataset(split_level, cv_partition)

    # specify the model name
    model_name = "esm2_t30_150M_UR50D"
    # model_name = "esm2_t33_650M_UR50D"
    # model_name = "esm2_t36_3B_UR50D"
    # model_name = "esm2_t48_15B_UR50D"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to("cuda")

    # Evaluate the model on the unsupervised contact map prediction task
    print("Evaluating ESM2 Model on Unsupervised Contact Map Prediction Task...")
    for split_level in split_levels:
        for cv_partition in cv_partitions:
            test(
                model, alphabet, split_level, cv_partition, n_bootstraps=10, n_samples=20
            )


if __name__ == "__main__":
    main(debug=True)
