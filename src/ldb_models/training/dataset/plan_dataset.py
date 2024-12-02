from typing import Tuple

from torch.utils.data import Dataset


class PlanDataset(Dataset):
    def __init__(self, plans, idxs):
        self.plans = plans
        self.idxs = [int(i) for i in idxs]
        assert len(self.plans) == len(self.idxs)

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, i: int):
        return self.idxs[i], self.plans[i]

    def split(self, ratio: float) -> Tuple:
        split_idx = int(len(self) * ratio)
        return PlanDataset(self.plans[:split_idx], self.idxs[:split_idx]), PlanDataset(self.plans[split_idx:], self.idxs[split_idx:])

    def remove_timeouts(self):
        self.plans = [plan for plan in self.plans if not plan['timeout']]
        self.idxs = [idx for idx, plan in zip(self.idxs, self.plans) if not plan['timeout']]
        return self
