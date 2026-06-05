from dataclasses import dataclass

from typing_extensions import Self

from atlasfold.common.metadata import (
    ExperimentRecord,
    JsonSerializable,
    Metadata,
    PredictionRecord,
)


@dataclass(slots=True)
class ComplexMetadata(JsonSerializable):
    id: str  # User/Author-defined name
    chains: list[Metadata]  # List of Metadata for each chain in the complex
    exp: ExperimentRecord | None = None  # Optional experimental record for the complex
    pred: PredictionRecord | None = None  # Optional prediction record for the complex
    cluster_id: str | None = None  # Optional cluster ID for the complex
    cluster_size: int | None = None  # Optional cluster size for the complex

    @property
    def num_chains(self) -> int:
        return len(self.chains)

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "chains": [chain.to_dict() for chain in self.chains],
        }
        # Add optional fields if they are not None
        for field in ["cluster_id", "cluster_size"]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field)
        # Add nested records if they are not None
        for field in ["exp", "pred"]:
            if getattr(self, field) is not None:
                data[field] = getattr(self, field).to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        chains = [Metadata.from_dict(chain_data) for chain_data in data["chains"]]

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
            chains=chains,
            cluster_id=data.get("cluster_id", None),
            cluster_size=data.get("cluster_size", None),
            exp=exp,
            pred=pred,
        )
