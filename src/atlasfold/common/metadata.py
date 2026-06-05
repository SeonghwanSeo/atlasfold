import dataclasses

from typing_extensions import Self

# For mmCIF parsing
# See mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html
CRYSTALLIZATION_METHODS = {
    "ELECTRON CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "POWDER DIFFRACTION",
    "X-RAY DIFFRACTION",
}
NMR_METHODS = {
    "SOLUTION NMR",
    "SOLID-STATE NMR",
}
EM_METHODS = {
    "ELECTRON MICROSCOPY",
}
OTHER_METHODS = {
    "FLUORESCENCE TRANSFER",
    "INFRARED SPECTROSCOPY",
    "SOLUTION SCATTERING",
}
ALL_EXPERIMENT_METHODS = (
    CRYSTALLIZATION_METHODS | NMR_METHODS | EM_METHODS | OTHER_METHODS
)


@dataclasses.dataclass(slots=True, kw_only=True)
class JsonSerializable:
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create ExperimentRecord from dictionary."""
        return cls(**data)


# TODO: do we need separate rcsb from experiment record?
@dataclasses.dataclass(slots=True, kw_only=True)
class ExperimentRecord(JsonSerializable):
    """Metadata record from RCSB PDB."""

    pdb_id: str
    release_date: str
    method: str
    resolution: float | None = None  # NMR: None

    @property
    def is_nmr_structure(self) -> bool:
        return self.method in NMR_METHODS

    @property
    def is_crystal_structure(self) -> bool:
        return self.method in CRYSTALLIZATION_METHODS

    @property
    def is_em_structure(self) -> bool:
        return self.method in EM_METHODS

    def __repr__(self) -> str:
        return (
            f"ExperimentRecord("
            f"pdb_id={self.pdb_id}, "
            f"method={self.method}, "
            f"resolution={self.resolution}, "
            ")"
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class PredictionRecord(JsonSerializable):
    """Metadata record from structure prediction."""

    # TODO: add more fields if necessary
    model: str | None = None  # e.g., "AlphaFold2"
    plddt: float | None = None


@dataclasses.dataclass(slots=True, kw_only=True)
class Metadata(JsonSerializable):
    id: str  # User/Author-defined name
    num_residues: int
    label_asym_id: str | None = None
    auth_asym_id: str | None = None
    entity_id: int | None = None  # starts from 1
    cluster_id: str | None = None
    cluster_size: int | None = None
    exp: ExperimentRecord | None = None
    pred: PredictionRecord | None = None

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "num_residues": self.num_residues,
        }
        # Add optional fields if they are not None
        for field in [
            "cluster_id",
            "cluster_size",
            "label_asym_id",
            "auth_asym_id",
            "entity_id",
        ]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field)
        # Add nested records if they are not None
        for field in ["exp", "pred"]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field).to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        # Create nested records if they are present
        if data.get("exp", None):
            exp = ExperimentRecord.from_dict(data["exp"])
        else:
            exp = None
        if data.get("pred", None):
            pred = PredictionRecord.from_dict(data["pred"])
        else:
            pred = None

        return cls(
            id=data["id"],
            num_residues=data["num_residues"],
            label_asym_id=data.get("label_asym_id", None),
            auth_asym_id=data.get("auth_asym_id", None),
            entity_id=data.get("entity_id", None),
            cluster_id=data.get("cluster_id", None),
            cluster_size=data.get("cluster_size", None),
            exp=exp,
            pred=pred,
        )
