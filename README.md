# faster_rcnn_online
- Original post: https://github.com/rbgirshick/py-faster-rcnn
# Modifications
- Online(with threading) image-batch, roidb, regression-target generator (see. Loader.py)
- Train/Valididation
# Usage
- Database path : /usrdata/ImageSearch/11st_DB/11st_All/Images/
- Annotation path : /usrdata/ImageSearch/11st_DB/11st_All/Annotations/
- train : annotations_train.txt
- validation : annotations_val.txt
- test : annotations_test.txt
- annotation format: image_name num_roi class x1 y1 x2 y2
- example (image_name : 12345.jpg, num_roi : 4)
-- annotation : 12345 4 cls x1 y1 x2 y2 cls x1 y1 x2 y2 cls x1 y1 x2 y2 cls x1 y1 x2 y2
- Prototxt path : /usrdata/ImageSearch/11st_DB/11st_All/prototxt/
- author: Park Jungyoung, Kim Moonki, Kim Taewan
