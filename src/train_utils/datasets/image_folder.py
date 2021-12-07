import numpy as np
import os
from torchvision.datasets import ImageFolder as TorchImageFolder
from typing import Optional
from torchvision.datasets.folder import find_classes


###############################################################################
# Due to a memory leak in pytorchs Dataloader when more then 1 worker is used #
# (https://github.com/pytorch/pytorch/issues/13246) a custom patch is applied #
# here. The code of "make_dataset" is unchanged from its implementation in    #
# torchvision.datasets.folder (torch==1.9.0) execpt from the commented parts  #
###############################################################################
def make_dataset(
    directory: str,
    class_to_idx: Optional['dict[str, int]'] = None,
    # the arguments "extensions" and "is_valid_file" are omitted
    # and the new argument "delimiter" is introduced
    delimiter='@',
    # The return type is changed to a list
) -> list:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of
    the ``find_classes`` function by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry")

    # Since the "extensions" and "is_valid_file" argument is omitted the
    # validity check for these variables has been removed.
    # Additionally the "is_valid_file" method was simpliefied and the valid
    # extensions are hardcoded
    def is_valid_file(filename: str) -> bool:
        return filename.lower().endswith(
            ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp')
        )

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    # The item changed from a tuple (path, class_index) to a
                    # concatenetd string e.g. "path@class_index"
                    item = f'{path}{delimiter}{class_index}'
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes

    # Since the "extensions" argument is omitted the validity check was
    # simplified here
    if empty_classes:
        raise FileNotFoundError(
            f"Found no valid file for the classes "
            f"{', '.join(sorted(empty_classes))}. "
        )

    # The string array is casted to a numpy square byte array to ensure only a
    # single refcount is used (see afore mentioned issue for details).
    instances_byte = np.array(instances).astype(np.string_)

    return instances_byte


###############################################################################
# To account for the changed "make_dataset" method the "__getitem__" method   #
# has to be adapted. Here is done for the "ImageFolder" class, the code is    #
# unchanged from its implementation in torchvision.datasets.folder            #
# (torch==1.9.0) execpt from the commented parts starting with two hash signs #
###############################################################################
class ImageFolder(TorchImageFolder):
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: 'dict[str, int]',
        ## the arguments "extensions" and "is_valid_file" are ignored since
        ## they are no longer used in the patched "make_dataset" method
        *_,
    ) -> 'list[str]':
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file
        instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to
                ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to
                class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed.
                    Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None
                or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to
            # idx logic of the find_classes() function, instead of using that
            # of the find_classes() method, which is potentially overridden and
            # thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )

        ## The patched "make_dataset" method is used and therefore the
        ## "extension" and "is_valid_file" arguments do not need to be passed
        return make_dataset(directory, class_to_idx)

    def __getitem__(self, index: int, delimiter='@') -> 'tuple[any, any]':
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the
                target class.
            delimiter: The delimiter to split target and path.
        """
        ## The target and sample extraction is adapted to work with
        ## a string seperated by a delimiter instead of a tuple.
        sample_string = str(self.samples[index], encoding='utf-8')
        path, target = sample_string.split(delimiter)
        target = int(target)

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
