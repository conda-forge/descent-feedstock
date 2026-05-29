"""Minimal smoke test for the ``descent`` conda-forge feedstock."""

from descent import __version__ as descent_version
from smee import __version__ as smee_version

assert descent_version != '0.0.0'
assert smee_version != '0.0.0'

import torch

from descent.targets.energy import Entry, create_dataset, extract_smiles, predict
from descent.train import ParameterConfig, Trainable

import openff.interchange
import openff.toolkit
import smee.converters

# Two-conformer water entry with arbitrary reference energies and forces.
entry: Entry = {
    "smiles": "[H:2][O:1][H:3]",
    "coords": torch.tensor(
        [
            [[0.0, 0.0, 0.0], [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0]],
            [[0.0, 0.0, 0.0], [-0.7, -0.5, 0.0], [0.7, -0.5, 0.0]],
        ]
    ),
    "energy": torch.tensor([2.0, 3.0]),
    "forces": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
}

dataset = create_dataset([entry])
assert len(dataset) == 1
assert extract_smiles(dataset) == [entry["smiles"]]

# Convert an OpenFF SMIRNOFF force field into a smee tensor force field.
force_field, [topology] = smee.converters.convert_interchange(
    openff.interchange.Interchange.from_smirnoff(
        openff.toolkit.ForceField("openff_unconstrained-2.3.0.offxml"),
        openff.toolkit.Molecule.from_mapped_smiles(entry["smiles"]).to_topology(),
    )
)

# Predict relative energies / forces for the dataset.
energy_ref, energy_pred, forces_ref, forces_pred = predict(
    dataset, force_field, {entry["smiles"]: topology}
)
assert energy_pred.shape == (2,)
assert forces_pred.shape == (6, 3)
assert torch.isfinite(energy_pred).all()
assert torch.isfinite(forces_pred).all()

# Use Trainable.
trainable = Trainable(
    force_field,
    parameters={"Bonds": ParameterConfig(cols=["k", "length"], scales={}, limits={})},
    attributes={},
)
values = trainable.to_values()
assert values.shape == (2,)
round_trip_ff = trainable.to_force_field(values)
assert round_trip_ff == force_field
