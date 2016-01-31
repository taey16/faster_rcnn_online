
from datasets.Loader import Loader

class eleven_all(Loader):

  def __init__(self, data_path, image_set):
    self._classes = ('__background__', # always index 0
      'bag', 'bra', 'jacket_coat', 'onepiece', 
      'pants', 'panty', 'shoes', 'skirt', 'swimwear', 
      'tshirts_shirts_blouse_hoody', 'vest', 'knit_cardigan')
    self.num_classes = 13
    self.name = 'eleven_all_' + image_set
    Loader.__init__(self, data_path, image_set)
