# coppied from https://github.com/pytorch/xla/blob/815197139b94e5655ed6b347f48864e73dc73011/torch_xla/utils/utils.py#L44
class SampleGenerator(object):
    """Iterator which returns multiple samples of a given input data.
    Can be used in place of a PyTorch `DataLoader` to generate synthetic data.
    Args:
    data: The data which should be returned at each iterator step.
    sample_count: The maximum number of `data` samples to be returned.
    """

    def __init__(self, data, sample_count):
        self._data = data
        self._sample_count = sample_count
        self._count = 0

    def __iter__(self):
        return SampleGenerator(self._data, self._sample_count)

    def __len__(self):
        return self._sample_count

    def __next__(self):
        return self.next()

    def next(self):
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data
