
from datasets.Loader import Loader

class eleven_all(Loader):

  def __init__(self, data_path, image_set):
    self._classes = ('__background__', # always index 0
                     'pant', 'leggings/stocking', 'hat/cap', 'shoe',
                     'blouse', 'jumper', 'panty', 'knit', 'swimsuit',
                     'shirt', 'coat', 'one-piece', 'tshirt', 'cardigan',
                     'skirt', 'jacket')
    self.num_classes = 17
    self.name = 'eleven_all_' + image_set
    Loader.__init__(self, data_path, image_set)
