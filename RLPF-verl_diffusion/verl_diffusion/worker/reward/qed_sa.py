from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors

from .base import BaseReward
from Model.EDM.qm9.analyze import check_stability
from Model.EDM.qm9.rdkit_functions import build_molecule
from verl_diffusion.protocol import DataProto, TensorDict


def _try_import_sa_scorer():
    try:
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        return sascorer
    except Exception:
        return None


class QEDSAReward(BaseReward):
    """Reward = w_qed * QED - w_sa * SA, with invalid-molecule penalty."""

    def __init__(
        self,
        dataset_info: dict,
        w_qed: float = 1.0,
        w_sa: float = 0.2,
        invalid_penalty: float = -1.0,
        stability_bonus: float = 0.1,
    ):
        super().__init__()
        self.dataset_info = dataset_info
        self.w_qed = float(w_qed)
        self.w_sa = float(w_sa)
        self.invalid_penalty = float(invalid_penalty)
        self.stability_bonus = float(stability_bonus)
        self.sa_scorer = _try_import_sa_scorer()

    def _heuristic_sa(self, mol: Chem.Mol) -> float:
        """Fallback SA proxy in [1, 10] if SA_Score contrib is unavailable."""
        mw = Descriptors.MolWt(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        raw = 1.5 + 0.003 * mw + 0.6 * n_rings + 0.2 * n_rot - 0.1 * n_hetero
        return float(np.clip(raw, 1.0, 10.0))

    def _sa(self, mol: Chem.Mol) -> float:
        if self.sa_scorer is not None:
            return float(self.sa_scorer.calculateScore(mol))
        return self._heuristic_sa(mol)

    def _extract_processed_list(self, samples: DataProto) -> list[tuple[torch.Tensor, torch.Tensor]]:
        one_hot = samples.batch["categorical"]
        x = samples.batch["x"]
        nodesxsample = samples.batch["nodesxsample"]
        processed = []
        for i in range(x.shape[0]):
            atom_type = one_hot[i].argmax(1).detach().cpu()
            pos = x[i].detach().cpu()
            n = int(nodesxsample[i])
            processed.append((pos[:n], atom_type[:n]))
        return processed

    def calculate_rewards(self, data: DataProto) -> DataProto:
        processed = self._extract_processed_list(data)
        rewards, qed_list, sa_list, stable_list, valid_list = [], [], [], [], []

        for pos, atom_type in processed:
            stable = float(check_stability(np.array(pos), atom_type.tolist(), self.dataset_info)[0])
            stable_list.append(stable)

            try:
                mol = build_molecule(pos, atom_type, self.dataset_info)
                Chem.SanitizeMol(mol)
                qed = float(QED.qed(mol))
                sa = self._sa(mol)
                reward = self.w_qed * qed - self.w_sa * sa + self.stability_bonus * stable
                valid = 1.0
            except Exception:
                qed = 0.0
                sa = 10.0
                reward = self.invalid_penalty
                valid = 0.0

            qed_list.append(qed)
            sa_list.append(sa)
            rewards.append(reward)
            valid_list.append(valid)

        result = {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "stability": torch.tensor(stable_list, dtype=torch.float32),
            "validity": torch.tensor(valid_list, dtype=torch.float32),
            "qed": torch.tensor(qed_list, dtype=torch.float32),
            "sa": torch.tensor(sa_list, dtype=torch.float32),
        }
        return DataProto(batch=TensorDict(result, batch_size=[len(rewards)]), meta_info=data.meta_info.copy())

