from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.utils import Dataset
from numpy import zeros
from numpy import asarray
import time
from os import listdir
from xml.etree import ElementTree


class MyMaskRCNNConfig(Config):
    NAME = "MaskRCNN_config"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes + BG
    NUM_CLASSES = 2 + 1

    STEPS_PER_EPOCH = 238

    LEARNING_RATE = 0.0005

    DETECTION_MIN_CONFIDENCE = 0.9

    MAX_GT_INSTANCES = 10


config = MyMaskRCNNConfig()

print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')

model.load_weights('mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])


class BlurAndScratchDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):

        self.add_class("dataset", 1, "blur")
        self.add_class("dataset", 2, "scratch")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in listdir(images_dir):

            image_id = filename[:-4]
            # after 120 if we are building the train set
            new_image_id = 0
            if image_id[0:4] == 'blur':
                new_image_id = image_id[4:]
            elif image_id[0:7] == 'scratch':
                new_image_id = image_id[7:]

            if is_train and int(new_image_id) >= 120:
                continue
            # before 120 if we are building the test/val set
            if not is_train and int(new_image_id) < 120:
                continue

            img_path = images_dir + filename

            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):

        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        path = info['annotation']

        boxes, w, h = self.extract_boxes(path)

        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            if info['id'][0:4] == 'blur':
                class_ids.append(self.class_names.index('blur'))
            elif info['id'][0:7] == 'scratch':
                class_ids.append(self.class_names.index('scratch'))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


train_set = BlurAndScratchDataset()
train_set.load_dataset('blur_and_scratch_dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

test_set = BlurAndScratchDataset()
test_set.load_dataset('blur_and_scratch_dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')
history = model.keras_model.history.history

model_path = 'mask_rcnn_' + '.' + str(time.time()) + '.h5'
model.keras_model.save_weights(model_path)
print("finish")


