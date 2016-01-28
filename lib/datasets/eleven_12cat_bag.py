
from datasets.Loader import Loader

class eleven_12cat_bag(Loader):

  def __init__(self, data_path, image_set):
    self.classes = ('__background__', 'bag')
    self.num_classes = 2
    self.name = 'eleven_12cat_bag_' + image_set
    Loader.__init__(self, data_path, image_set)
    

"""
#annotation_filename = '/storage/product/detection/11st_Bag/Annotations/annotations_train.txt'
image_path_prefix = '/storage/product/detection/11st_Bag'
import pdb; pdb.set_trace()
dataloader = eleven_12cat_bag(image_path_prefix, 'train')
"""
