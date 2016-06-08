
from datasets.Loader import Loader

class eleven_19_category(Loader):

  def __init__(self, data_path, image_set):
    self._classes = ('__background__', # always index 0
      'tshirts', 'shirts', 'blouse', 'knit', 
      'jacket', 'onepiece', 'skirt', 'coat', 'cardigan', 
      'vest', 'pants', 'leggings', 'shoes', 'bag', 'swimwear', 'hat', 'panties', 'bra', 'socks')
    self.num_classes = 20
    self.name = 'eleven_19_category_' + image_set
    Loader.__init__(self, data_path, image_set)
