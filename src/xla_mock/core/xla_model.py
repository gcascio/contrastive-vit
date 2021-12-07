import torch


class RateTracker:
    def add(self, *_):
        pass

    def rate(self):
        return 0

    def global_rate(self):
        return 0


def optimizer_step(optimizer):
    optimizer.step()


def get_ordinal():
    return 0


def xrt_world_size():
    return 1


def xla_device():
    return 'cpu'


def master_print(args):
    print(args)


def is_master_ordinal():
    return True


def save(*args, **kwargs):
    torch.save(*args, **kwargs)


def set_rng_seed(*_):
    pass


def rendezvous(*_):
    pass
