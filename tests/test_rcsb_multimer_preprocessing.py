import importlib.util
import logging
import pathlib
import sys
import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import numpy as np

from atlasfold.common import metadata, protein
from atlasfold.train.multimer.dataset import (
    LMDBDataset,
    MultimerDataPipeline,
    RCSBTrainingDataset,
    TrainingDataset,
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


B1_PROCESS = load_script(
    "test_rcsb_multimer_b1_process",
    "scripts/preprocess_multimer/rcsb/b1_process.py",
)
A2_CLUSTER = load_script(
    "test_rcsb_multimer_a2_cluster",
    "scripts/preprocess_multimer/rcsb/a2_cluster.py",
)
C3_TEMPLATE_MAPPING = load_script(
    "test_rcsb_multimer_c3_template_mapping",
    "scripts/preprocess_multimer/rcsb/c3_create_template_mapping.py",
)


class MultimerSerializationTest(unittest.TestCase):
    def test_round_trip_preserves_multimer_ids(self):
        compl = protein.ProteinMultimer(
            name="complex",
            chains=[
                protein.Protein.get_empty("a", "AAAA"),
                protein.Protein.get_empty("b", "CCCC"),
            ],
            entity_ids=[7, 9],
            asym_ids=[4, 8],
            sym_ids=[2, 3],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "complex.npz"
            MultimerDataPipeline.save(compl, path)
            loaded = MultimerDataPipeline.load(path)

        self.assertEqual(loaded.entity_ids, [7, 9])
        self.assertEqual(loaded.asym_ids, [4, 8])
        self.assertEqual(loaded.sym_ids, [2, 3])

    def test_legacy_npz_without_ids_still_loads(self):
        compl = protein.ProteinMultimer(
            name="legacy",
            chains=[
                protein.Protein.get_empty("a", "AAAA"),
                protein.Protein.get_empty("b", "AAAA"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            new_path = pathlib.Path(tmp_dir) / "new.npz"
            legacy_path = pathlib.Path(tmp_dir) / "legacy.npz"
            MultimerDataPipeline.save(compl, new_path)
            with np.load(new_path) as data:
                legacy_data = {
                    key: data[key]
                    for key in data.files
                    if key not in {"entity_ids", "asym_ids", "sym_ids"}
                }
            np.savez_compressed(legacy_path, **legacy_data)
            loaded = MultimerDataPipeline.load(legacy_path)

        self.assertEqual(loaded.entity_ids, [1, 1])
        self.assertEqual(loaded.asym_ids, [1, 2])
        self.assertEqual(loaded.sym_ids, [1, 2])


class RCSBSamplingTest(unittest.TestCase):
    def test_all_assembly_weight_scales_sampling_probability(self):
        metadatas = []
        for name, num_assemblies in [("1abc-assembly1", 2), ("2abc", 1)]:
            metadatas.append(
                {
                    "id": name,
                    "num_assemblies": num_assemblies,
                    "assembly_sampling_weight": 1.0 / num_assemblies,
                    "chains": [{"cluster_size": 1}],
                    "interfaces": [],
                }
            )

        def fake_training_init(dataset, *args, **kwargs):
            dataset.metadatas = metadatas

        config = SimpleNamespace(is_multimer=True)
        with mock.patch.object(TrainingDataset, "__init__", fake_training_init):
            dataset = RCSBTrainingDataset(config)

        np.testing.assert_allclose(dataset.weights, [0.5, 1.0])

    def test_assembly_weight_is_derived_from_assembly_count(self):
        self.assertEqual(
            RCSBTrainingDataset.get_assembly_sampling_weight(
                {"id": "1abc-assembly2", "num_assemblies": 4}
            ),
            0.25,
        )

    def test_inconsistent_assembly_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "1 / num_assemblies"):
            RCSBTrainingDataset.get_assembly_sampling_weight(
                {
                    "id": "1abc-assembly2",
                    "num_assemblies": 4,
                    "assembly_sampling_weight": 0.5,
                }
            )


class AssemblyOutputNamingTest(unittest.TestCase):
    def test_all_assemblies_uses_separate_dataset_directory(self):
        train_filter = B1_PROCESS.AF3_SPLITS["train"]

        self.assertEqual(
            B1_PROCESS.get_dataset_name(train_filter, all_assemblies=False),
            "rcsb_multimer",
        )
        self.assertEqual(
            B1_PROCESS.get_dataset_name(train_filter, all_assemblies=True),
            "rcsb_multimer_assembly",
        )


class StructureFilteringTest(unittest.TestCase):
    def test_sequence_composition_filter_is_complex_level(self):
        poly_a = protein.Protein.get_empty("poly-a", "AAAAAAAAAA")
        diverse = protein.Protein.get_empty("diverse", "CDEFGHIKLM")

        self.assertFalse(B1_PROCESS.validate_complex_sequence([poly_a]))
        self.assertTrue(B1_PROCESS.validate_complex_sequence([poly_a, diverse]))

    def test_short_chain_filter_is_independent_of_composition(self):
        self.assertFalse(
            B1_PROCESS.validate_chain_length(protein.Protein.get_empty("short", "AAA"))
        )
        self.assertTrue(
            B1_PROCESS.validate_chain_length(
                protein.Protein.get_empty("long-enough", "AAAA")
            )
        )

    def test_unresolved_gap_does_not_create_ca_jump(self):
        prot = protein.Protein.get_empty("gap", "AAAAAA")
        prot.coordinates[0, 1] = [0.0, 0.0, 0.0]
        prot.coordinates[1, 1] = [3.8, 0.0, 0.0]
        prot.coordinates[4, 1] = [30.0, 0.0, 0.0]
        prot.coordinates[5, 1] = [33.8, 0.0, 0.0]

        self.assertTrue(B1_PROCESS.validate_geometry(prot))

    def test_resolved_adjacent_ca_jump_is_rejected(self):
        prot = protein.Protein.get_empty("jump", "AAAA")
        prot.coordinates[:, 1, :] = [
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [23.8, 0.0, 0.0],
        ]

        self.assertFalse(B1_PROCESS.validate_geometry(prot))

    def test_equal_clash_ratio_removes_smaller_chain(self):
        def make_chain(name: str, num_atoms: int) -> protein.Protein:
            chain = protein.Protein.get_empty(name, "AAAA")
            chain.coordinates.reshape(-1, 3)[:num_atoms] = 0.0
            return chain

        large = make_chain("large", 10)
        small = make_chain("small", 5)
        metadatas = [
            metadata.Metadata(
                id="large",
                num_residues=4,
                entity_id=1,
                asym_id=1,
                sym_id=1,
            ),
            metadata.Metadata(
                id="small",
                num_residues=4,
                entity_id=2,
                asym_id=2,
                sym_id=1,
            ),
        ]

        kept, _ = B1_PROCESS.filter_clashing_chains([large, small], metadatas)

        self.assertEqual([chain.name for chain in kept], ["large"])

    def test_clash_statistics_report_removed_chain(self):
        def make_chain(name: str, num_atoms: int) -> protein.Protein:
            chain = protein.Protein.get_empty(name, "AAAA")
            chain.coordinates.reshape(-1, 3)[:num_atoms] = 0.0
            return chain

        metadatas = [
            metadata.Metadata(
                id=name,
                num_residues=4,
                entity_id=index,
                asym_id=index,
                sym_id=1,
            )
            for index, name in enumerate(["large", "small"], 1)
        ]
        _, _, stats = B1_PROCESS.filter_clashing_chains_with_stats(
            [make_chain("large", 10), make_chain("small", 5)],
            metadatas,
        )

        self.assertEqual(stats.contact_chain_pairs, 1)
        self.assertEqual(stats.atom_clash_chain_pairs, 1)
        self.assertEqual(stats.severe_clash_chain_pairs, 1)
        self.assertEqual(stats.chains_removed, 1)


class SequenceClusteringTest(unittest.TestCase):
    def test_short_proteins_use_exact_identity_clusters(self):
        sequences = [
            ("1abc_1", "KFK"),
            ("1abc_2", "KYK"),
            ("2abc_1", "KFK"),
            ("3abc_1", "ACDEFGHIKL"),
            ("4abc_1", "ACDEFGHIKM"),
        ]
        observed_long_sequences = []

        def fake_mmseqs(long_sequences, **kwargs):
            observed_long_sequences.extend(long_sequences)
            return {seq_id: "long-cluster" for seq_id, _ in long_sequences}

        with mock.patch.object(A2_CLUSTER, "run_mmseqs2_cluster", fake_mmseqs):
            clusters = A2_CLUSTER.cluster_sequences(sequences, "mmseqs")

        self.assertEqual(clusters["1abc_1"], clusters["2abc_1"])
        self.assertNotEqual(clusters["1abc_1"], clusters["1abc_2"])
        self.assertEqual(
            {sequence for _, sequence in observed_long_sequences},
            {"ACDEFGHIKL", "ACDEFGHIKM"},
        )


class TemplatePipelineTest(unittest.TestCase):
    @staticmethod
    def write_mapping(path: pathlib.Path, idx_map: np.ndarray):
        info = {
            "idx_map": idx_map,
            "release_date": "2020-01-01",
            "index": 0,
        }
        np.savez(path, hit=np.asarray(info, dtype=object))

    def test_all_assembly_template_key_falls_back_to_base_pdb(self):
        self.assertEqual(
            LMDBDataset.get_base_template_mapping_key("1abc-assembly12_3"),
            "1abc_3",
        )
        self.assertIsNone(LMDBDataset.get_base_template_mapping_key("1abc_3"))

    def test_empty_template_alignment_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "query.npz"
            self.write_mapping(path, np.empty((0, 2), dtype=np.int64))
            record = C3_TEMPLATE_MAPPING.load_entry_mapping(
                path,
                datetime.fromisoformat("2021-01-01"),
            )

        self.assertEqual(record["templates"], [])

    def test_negative_template_index_is_rejected_before_cast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "query.npz"
            self.write_mapping(path, np.asarray([[-1, 2]], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "expected to be 1-based"):
                C3_TEMPLATE_MAPPING.load_entry_mapping(
                    path,
                    datetime.fromisoformat("2021-01-01"),
                )

    def test_missing_template_hit_is_backfilled(self):
        dataset = LMDBDataset.__new__(LMDBDataset)
        dataset.max_templates = 2
        dataset.logger = logging.getLogger("template-backfill-test")
        query = protein.Protein.get_empty("query", "AAAA")
        templates = {
            "valid-1": protein.Protein.get_empty("valid-1", "AAAA"),
            "valid-2": protein.Protein.get_empty("valid-2", "AAAA"),
        }
        hits = []
        for index, template_id in enumerate(["missing", "valid-1", "valid-2"]):
            hits.append(
                {
                    "template_id": template_id,
                    "index": index,
                    "entry_indices": np.arange(1, 5),
                    "template_indices": np.arange(1, 5),
                }
            )
        dataset.fetch_template_hits = lambda _: hits

        def fetch_template(template_id: str):
            if template_id not in templates:
                raise KeyError(template_id)
            return templates[template_id]

        dataset.fetch_template = fetch_template

        features = dataset.prepare_chain_template_inputs(query, "query_1")

        np.testing.assert_array_equal(features["template.mask"], [True, True])

    def test_training_template_selection_backfills_invalid_candidates(self):
        class FixedRng:
            @staticmethod
            def random():
                return 0.0

            @staticmethod
            def integers(*args, **kwargs):
                return 2

            @staticmethod
            def permutation(length):
                return np.arange(length)

        dataset = TrainingDataset.__new__(TrainingDataset)
        dataset.max_templates = 2
        dataset.template_prob = 1.0
        dataset.logger = logging.getLogger("training-template-backfill-test")
        query = protein.Protein.get_empty("query", "AAAA")
        templates = {
            "valid-1": protein.Protein.get_empty("valid-1", "AAAA"),
            "valid-2": protein.Protein.get_empty("valid-2", "AAAA"),
        }
        hits = [
            {
                "template_id": template_id,
                "index": index,
                "entry_indices": np.arange(1, 5),
                "template_indices": np.arange(1, 5),
            }
            for index, template_id in enumerate(["missing", "valid-1", "valid-2"])
        ]
        dataset.fetch_template_hits = lambda _: hits

        def fetch_template(template_id: str):
            if template_id not in templates:
                raise KeyError(template_id)
            return templates[template_id]

        dataset.fetch_template = fetch_template

        features = dataset.prepare_chain_template_inputs(
            query,
            "query_1",
            FixedRng(),
        )

        np.testing.assert_array_equal(features["template.mask"], [True, True])

    def test_all_assembly_mapping_reuses_base_pdb_template_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)
            source_path = tmp_path / "1abc_1.npz"
            source_path.touch()
            chain_mapping = {
                "1abc-assembly2_A1": (
                    "1abc-assembly2_1",
                    "1abc_1",
                    20,
                    datetime.fromisoformat("2021-01-01"),
                )
            }

            selected, missing = C3_TEMPLATE_MAPPING.select_mapping_paths(
                [source_path],
                chain_mapping,
            )

        self.assertEqual(missing, 0)
        self.assertEqual(selected[0][0], source_path)
        self.assertEqual(selected[0][1], "1abc-assembly2_1")


if __name__ == "__main__":
    unittest.main()
