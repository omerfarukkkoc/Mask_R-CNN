from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn.utils import Dataset
from mrcnn import visualize
from os import listdir


class MyMaskRCNNConfig(Config):
    NAME = "MaskRCNN_config"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes + BG
    NUM_CLASSES = 2 + 1

    DETECTION_MIN_CONFIDENCE = 0.9

    MAX_GT_INSTANCES = 10


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

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


test_set = BlurAndScratchDataset()
test_set.load_dataset('blur_and_scratch_dataset', is_train=False)
test_set.prepare()

model_path = 'mask_rcnn_.1576924943.852383.h5'
config = MyMaskRCNNConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
model.load_weights(model_path, by_name=True)
image_id = 4
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, config, image_id,
                                                                          use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
results = model.detect([image], verbose=1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            test_set.class_names, r['scores'],
                            title="Predictions")
