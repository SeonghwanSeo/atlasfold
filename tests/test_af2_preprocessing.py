import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np

from atlasfold.common import protein
from atlasfold.train.monomer.dataset import (
    DataPipeline,
    TrainingDataset,
    TrainingDatasetConfig,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_script(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


A2_CONSTRUCT_LMDB = load_script(
    "test_af2_a2_construct_lmdb",
    "scripts/preprocess/af2/a2_construct_lmdb.py",
)


class AF2PreprocessingTest(unittest.TestCase):
    def test_process_metadata_loads_data_pipeline_npz(self):
        prot = protein.Protein.get_empty("af2-entry", "AAAA")
        prot.b_factors[:, 1] = [90.0, 80.0, 70.0, 60.0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "af2-entry.npz"
            DataPipeline.save(prot, path)
            entry_id, metadata_dict, success = A2_CONSTRUCT_LMDB.process_metadata(path)

        self.assertTrue(success)
        self.assertEqual(entry_id, "af2-entry")
        self.assertIsNotNone(metadata_dict)
        self.assertEqual(metadata_dict["num_residues"], 4)
        self.assertEqual(metadata_dict["pred"]["plddt"], 75.0)

    def test_distillation_labels_mask_low_confidence_residues(self):
        prot = protein.Protein.create(
            name="af2-entry",
            sequence="AAA",
            coordinates=np.zeros((3, 14, 3), dtype=np.float32),
            b_factors=np.broadcast_to(
                np.asarray([49.0, 50.0, 51.0], dtype=np.float32)[:, None],
                (3, 14),
            ).copy(),
        )
        dataset = TrainingDataset.__new__(TrainingDataset)
        dataset.config = TrainingDatasetConfig(
            name="af2",
            weight=1.0,
            is_distillation=True,
            residue_plddt_threshold=50.0,
        )

        labels = dataset.prepare_labels(prot)

        self.assertFalse(labels["resolved_mask"][0].any())
        self.assertFalse(labels["resolved_mask"][1].any())
        self.assertTrue(labels["resolved_mask"][2].all())
        self.assertTrue(np.all(labels["coordinates"][:2] == 0.0))

    def test_confidence_threshold_does_not_mask_experimental_labels(self):
        prot = protein.Protein.create(
            name="rcsb-entry",
            sequence="A",
            coordinates=np.ones((1, 14, 3), dtype=np.float32),
            b_factors=np.zeros((1, 14), dtype=np.float32),
        )
        dataset = TrainingDataset.__new__(TrainingDataset)
        dataset.config = TrainingDatasetConfig(
            name="rcsb",
            weight=1.0,
            is_distillation=False,
            residue_plddt_threshold=50.0,
        )

        labels = dataset.prepare_labels(prot)

        self.assertTrue(labels["resolved_mask"].all())


if __name__ == "__main__":
    unittest.main()
