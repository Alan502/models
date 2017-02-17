from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from inception.dataset import Dataset

class PosesData(Dataset):
  """Poses data set."""

  def __init__(self, subset):
    super(PosesData, self).__init__('Poses', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 19

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 14467
    if self.subset == 'validation':
      return 3563

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""
    print('Downloaded the dataset for the poses data.')
