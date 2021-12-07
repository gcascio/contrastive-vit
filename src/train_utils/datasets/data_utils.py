import math
import random
import torch
import torch_xla.core.xla_model as xm


def get_subset(subset: float, dataset):
    train_dataset_len = len(dataset)
    subset_len = math.floor(train_dataset_len * subset)
    indices = random.sample(range(train_dataset_len), subset_len)
    return torch.utils.data.Subset(dataset, indices)


def get_data_loader(
    train_dataset,
    test_dataset,
    batch_size,
    test_set_batch_size=None,
    num_workers=1,
    drop_last=False,
    subset: float = None,
):
    train_sampler, test_sampler = None, None

    if subset:
        train_dataset = get_subset(subset, train_dataset)

    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=drop_last,
        shuffle=False if train_sampler else True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_set_batch_size or batch_size,
        sampler=test_sampler,
        drop_last=drop_last,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader
