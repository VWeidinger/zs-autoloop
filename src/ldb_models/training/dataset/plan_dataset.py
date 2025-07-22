import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from training.training.inference import inference


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
        return PlanDataset(self.plans[:split_idx], self.idxs[:split_idx]), PlanDataset(
            self.plans[split_idx:], self.idxs[split_idx:]
        )

    def remove_timeouts(self):
        self.plans = [plan for plan in self.plans if not plan["timeout"]]
        self.idxs = [
            idx for idx, plan in zip(self.idxs, self.plans) if not plan["timeout"]
        ]
        return self


def serialize(data):
    buffer = pickle.dumps(data, protocol=-1)
    return np.frombuffer(buffer, dtype=np.uint8)


class PreBatchedDataset(Dataset):
    """
    Serialized Dataset to prevent python objects copy-on-write behavior in shared memory for
    dataloader workers. That is, as python objects use reference counters, they get coppied in
    memory as soon as a worker tries to acces them. Thus each worker starts replicating the dataset
    in its own/private memory. Tensors prevent this, as they implement their own pickler.
        - GitHub Issue: https://github.com/pytorch/pytorch/issues/13246
        - Based on: https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader
        - Code on GitHub: https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/main/serialize.py
    """

    def __init__(self, batches: List, batch_size: int):
        self.batch_size = batch_size

        # serialize batches
        self._batches = [serialize(batch) for batch in batches]
        self._addr = torch.asarray(
            [len(batch) for batch in self._batches], dtype=torch.int64
        )
        self._addr = torch.cumsum(self._addr, dim=0)
        self._batches = np.concatenate(self._batches)

        # convert to tensor
        self._batches = torch.from_numpy(self._batches)

        print(
            f"Serialization done, dataset takes {len(self._batches) / 1024**2:.2f} MiB",
            flush=True,
        )

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        # lookup start / end adress
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()

        # get batch
        bytes = memoryview(self._batches[start_addr:end_addr].numpy())
        batch = pickle.loads(bytes)
        return batch


class NonSerializedPreBatchedDataset(Dataset):
    def __init__(self, batches: List, batch_size: int):
        self.batches = batches
        self.batch_size = batch_size

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class OHEMPlanDataset(Dataset):
    """
    Online-Hard-Example-Mining Dataset
    """

    def __init__(
        self,
        plan_lists: List[List],
        plan_idxs_lists: List[List],
        workload_file_names: List[Path] = None,
        limit_plans_per_file: int = None,
        limit_plans_overall: int = None,
        slice_by: str = "file",
    ):
        """
        Args:
            plan_lists: List of list of plans (per file)
            plan_idxs_lists: List of list of indexes (per file)
            limit_plans_per_file: Limit number of plans per file
            limit_plans_overall: Limit number of plans overall
        """

        assert slice_by in ["file", "dataset", "none"], f"Invalid slice_by={slice_by}!"

        # store input
        self.plan_lists = plan_lists
        self.plan_idxs_lists = plan_idxs_lists
        self.limit_plans_per_file = limit_plans_per_file
        self.limit_plans_overall = limit_plans_overall
        self.slice_by = slice_by
        self.workload_file_names = workload_file_names

        if self.limit_plans_per_file is not None:
            assert isinstance(self.limit_plans_per_file, int)

        # initialize placeholders - will be set by deferred setup
        self.model = None
        self.criterion = None
        self.dataloader_args = None
        self.assembled_plan_list = None
        self.assembled_idx_list = None

        # run checks
        assert len(self.plan_lists) == len(self.plan_idxs_lists)
        for l1, l2 in zip(self.plan_lists, self.plan_idxs_lists):
            assert len(l1) == len(l2)

        # count total number of plans
        self.total_num_plans = sum([len(l) for l in self.plan_lists])
        print(
            f"OHEMDataset initialized with {self.total_num_plans} plans - willl pick {self.__len__()} plans from it (limit_plans_per_file={self.limit_plans_per_file}, limit_plans_overall={self.limit_plans_overall})"
        )

        # apply ohem:
        self.apply_ohem = (
            limit_plans_per_file is not None or limit_plans_overall is not None
        )  # apply OHEM if any limit is set - if not limit is given, always return everything -> no OHEM necessary
        self.apply_ohem = self.apply_ohem and self.total_num_plans > self.__len__()

        if not self.apply_ohem:
            print(
                f"Disable OHEM - target dataset size ({self.__len__()}) is larger than num available plans ({self.total_num_plans})."
            )

        # initialize plan difficulty
        self.losses = torch.ones(self.total_num_plans) * float(
            "inf"
        )  # initialize with high loss

        # merge plan lists
        self.merge_plan_lists_according_to_slicing()

        if self.apply_ohem:
            # create placeholders
            self._all_plans_dataset = None
            self._all_plans_dataloader = None
        else:
            # still run this - to make sure assembled_plan_list is set
            self._assemble_plan_list_from_files_with_limits_and_difficulty(
                skip_sort=True
            )

    def merge_plan_lists_according_to_slicing(self):
        """
        Merge plan lists according to slicing
        """
        if self.slice_by == "file":
            # keep as is
            pass
        elif self.slice_by == "dataset":
            # merge all plans which are of the same dataset
            file_data_dict = defaultdict(list)

            for wl_name, plans, plan_idxs in zip(
                self.workload_file_names, self.plan_lists, self.plan_idxs_lists
            ):
                # extract dataset from wl_name
                dataset_name = wl_name.parent.name
                file_data_dict[dataset_name].append((wl_name, plans, plan_idxs))

            # merge all plans
            plan_slices = []
            plan_idxs_slices = []

            for dataset_name, data_lists in file_data_dict.items():
                all_plans = [p for _, plans, _ in data_lists for p in plans]
                all_plan_idxs = [p for _, _, plan_idxs in data_lists for p in plan_idxs]

                plan_slices.append(all_plans)
                plan_idxs_slices.append(all_plan_idxs)

            self.plan_lists = plan_slices
            self.plan_idxs_lists = plan_idxs_slices

        elif self.slice_by == "none":
            # merge all plans
            all_plans = [p for plans in self.plan_lists for p in plans]
            all_plan_idxs = [p for plan_idxs in self.plan_idxs_lists for p in plan_idxs]

            self.plan_lists = [all_plans]
            self.plan_idxs_lists = [all_plan_idxs]

    def deferred_setup(self, model, criterion, initialize_difficulty: bool = False):
        """
        Deferred setup - initialize all plans dataset
        """
        self.model = model
        self.criterion = criterion

        if self.apply_ohem:
            ###
            # initialize all plans dataset
            ###

            # concat all plan lists
            unified_p_list = [plan for plans in self.plan_lists for plan in plans]

            assert self.dataloader_args is not None, "dataloader_args not set!"

            # create plan dataset & dataloader - this is only internal for OHEM!
            self._all_plans_dataset = PlanDataset(
                unified_p_list, range(self.total_num_plans)
            )
            self._all_plans_dataloader = DataLoader(
                self._all_plans_dataset, **self.dataloader_args
            )

            if initialize_difficulty:
                # run model forward pass
                self.update_difficulty()
            else:
                # assemble plan list
                self._assemble_plan_list_from_files_with_limits_and_difficulty(
                    skip_sort=self.apply_ohem
                )

    def update_difficulty(self):
        """
        Run forward pass and calculate loss for all plans
        """
        print(f"Update difficulty for {self.total_num_plans} plans...")
        if self.apply_ohem:
            assert self.__len__() < self.total_num_plans, (
                f"len(self)={self.__len__()} is larger than self.total_num_plans={self.total_num_plans}. In this case OHEM makes no sense."
            )

            self.model.eval()  # set model to eval mode

            # run inference
            labels, preds = inference(
                dataloader=self._all_plans_dataloader, model=self.model
            )

            # compute loss/error
            self.losses: torch.Tensor = self.criterion(preds, labels).cpu()

            # compute overall mean loss
            mean_loss = torch.mean(self.losses)
            print(f"Mean loss: {mean_loss:.4f}")

            # assemble plan list from files with limits and difficulty
            self._assemble_plan_list_from_files_with_limits_and_difficulty()

    def __len__(self):
        # apply limits

        dataset_length = 0

        # apply limit per file
        for plans in self.plan_lists:
            if self.limit_plans_per_file is not None:
                dataset_length += min(len(plans), self.limit_plans_per_file)
            else:
                dataset_length += len(plans)

        # apply limit overall
        if self.limit_plans_overall is not None:
            dataset_length = min(dataset_length, self.limit_plans_overall)

        return dataset_length

    def _assemble_plan_list_from_files_with_limits_and_difficulty(
        self, skip_sort: bool = False
    ):
        """
        Assemble plan list from files with limits and difficulty
        """
        # assemble plan list
        plan_list: List = []
        idx_list: List[int] = []

        plan_count = 0
        for plans, plan_idxs in zip(self.plan_lists, self.plan_idxs_lists):
            if skip_sort:
                # do not sort plans
                sorted_plans = plans
                sorted_plan_idxs = plan_idxs
            else:
                # extract corresponding losses
                loss_list = self.losses[plan_count : plan_count + len(plans)]

                # sort plans by loss (descending)
                sorted_idxs = torch.argsort(loss_list, descending=True)
                sorted_plans = [plans[i] for i in sorted_idxs]
                sorted_plan_idxs = [plan_idxs[i] for i in sorted_idxs]

            if self.limit_plans_per_file is not None:
                assert isinstance(self.limit_plans_per_file, int)
                plan_list.extend(sorted_plans[: self.limit_plans_per_file])
                idx_list.extend(sorted_plan_idxs[: self.limit_plans_per_file])
            else:
                plan_list.extend(plans)
                idx_list.extend(sorted_plan_idxs)

            # update plan count
            plan_count += len(plans)

        # apply limit overall
        if self.limit_plans_overall is not None:
            # apply shuffle with fixed seed
            random = np.random.RandomState(42)

            # shuffle plan_list and idx_list similarly
            shuffle_idxs = np.arange(len(plan_list))
            random.shuffle(shuffle_idxs)

            # apply shuffle on plan_list and idx_list (using numpy)
            plan_list = [plan_list[i] for i in shuffle_idxs]
            idx_list = [idx_list[i] for i in shuffle_idxs]

            # cap to limit
            plan_list = plan_list[: self.limit_plans_overall]
            idx_list = idx_list[: self.limit_plans_overall]

        # randomly shuffle plans and idxs
        random = np.random.RandomState(42)
        shuffle_idxs = np.arange(len(plan_list))
        plan_list = [plan_list[i] for i in shuffle_idxs]
        idx_list = [idx_list[i] for i in shuffle_idxs]

        self.assembled_plan_list = plan_list
        self.assembled_idx_list = idx_list

    def __getitem__(self, i: int):
        assert self.assembled_plan_list is not None, (
            "assembled_plan_list not set! Have you called deferred_setup()?"
        )
        return self.assembled_idx_list[i], self.assembled_plan_list[i]
