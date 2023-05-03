# Description: This file is used to train the Mask R-CNN model on the coco formatted cuts.
# Command Line Arguments: --data_path, --weights
# defaults: data_path = '../data/coco_formatted_cuts_split', weights = '../data/weights/mask_rcnn_coco.h5'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
import skimage


from PIL import Image, ImageDraw

workspace_dir = str(Path(__file__).parent.parent)
sys.path.insert(1, f'{workspace_dir}/Mask_RCNN')

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

ROOT_DIR = os.path.abspath(".") # Root directory of the project
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "data/weights/mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class CutsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cuts"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 2  # background + cut
    # (COW, PIG, PLATE)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(CutsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class CutsDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    # TEMPORARY: Will be done with a command line argument
    data_path = None

    def load_cuts(self, split = "train"):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        
        assert split in ["train", "test", "val"]
        
        # Add classes
        self.add_class("cuts", 1, "base")
        
        # Create file tuple with all of the files ordered alphabetically. We use this for file
        # indexes
        self.file_list = [f.replace(".json", "") for f in os.listdir(os.path.join(self.data_path, split)) if os.path.isfile(os.path.join(self.data_path, split, f)) and ".json" in f]
        self.file_list = tuple(self.file_list)

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for img_id in self.file_list:
            json_path = os.path.join(self.data_path, split, img_id + ".json")
            
            with open(json_path) as user_file:
                json_data = json.load(user_file)
                
            # Get Height and Width of image.
            if os.path.isfile(os.path.join(self.data_path, split, img_id + ".jpg")):
                img_path = os.path.join(self.data_path, split, img_id + ".jpg")
                with Image.open(img_path) as im:
                    width, height = im.size
            elif os.path.isfile(os.path.join(self.data_path, split, img_id + ".JPG")):
                img_path = os.path.join(self.data_path, split, img_id + ".JPG")
                with Image.open(img_path) as im:
                    width, height = im.size

            # Adding only the base cuts or single number labelled images
            shape_types_list = [shape['label'] for shape in json_data['shapes']]
            single_number_list = [shape['label'].isdigit() for shape in json_data['shapes']]

            if 'base' not in shape_types_list and True not in single_number_list:
                continue

            if True in single_number_list:
                polygons = json_data['shapes'][single_number_list.index(True)]
                polygons['label'] = 'base'
            else:
                polygons = json_data['shapes'][shape_types_list.index('base')]
            
            # Add image to the dataset
            self.add_image("cuts", image_id = img_id, json_path = json_path,
                           path = img_path, width = width, height = height, 
                           polygons = [polygons])

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        img_path = info["path"]
        with Image.open(img_path) as img:
            image = np.array(img)

        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cuts":
            return info["cuts"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        polygons = info["polygons"]
        
        count = len(polygons)
        
        mask_list = list()
        class_ids = list()
        
        for i, instance in enumerate(polygons):
            label = instance["label"]
            points = instance["points"]
            points = [tuple(point) for point in points]
            
            label_index = self.class_names.index(label)
            
            img = Image.new('L', (info['width'], info['height']), 0)
            ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
            mask = np.array(img)
            
            # TODO: FIX THIS TO TURN (3, 200, 300) -> (200, 300, 3) for exmaple
            mask.astype(np.bool)
            
            mask_list.append(mask)
            class_ids.append(label_index)
        
        mask_list = np.array(mask_list).astype(np.bool)
        #print(mask_list, class_ids)
        mask_list = np.transpose(mask_list, [1, 2, 0])
        class_ids = np.array(class_ids).astype(np.int32)
        
        return mask_list.astype(np.bool), class_ids.astype(np.int32)

def predict_file(image_path, save_path, model):
    # Take datapath and load it as a numpy array
    image = skimage.io.imread(image_path)

    results = model.detect([image], verbose=1)
    r = results[0]
    returned_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            ["BG", "base"], r['scores'], ax=get_ax(), return_image = True, show_bbox = False)

    # Save the image as a png in the save path
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, image_path.split("/")[-1])
    skimage.io.imsave(save_path, returned_image)

# Goes through direcotry and predicts all images in the directory
def predict_folder(data_path, save_path, model):
    
    # Get all the images in the directory
    image_list = [os.path.join(data_path, image) for image in os.listdir(data_path) if image.endswith(".jpg") or image.endswith(".JPG")]

    for image_path in image_list:
        predict_file(image_path, save_path, model)

def predict(data_path, save_path, model_path):

    config = InferenceConfig()

    model = load_model(model_path, config)

    # Checking to see if file data path is an image or a directory
    if os.path.isfile(data_path):
        predict_file(data_path, save_path, model)
    elif os.path.isdir(data_path):
        predict_folder()

#####################
# Utility Functions #
#####################

def load_model(init_with : str, config : InferenceConfig):
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)

    if os.path.isfile(init_with):
        path = init_with
        tf.keras.Model.load_weights(model.keras_model, path, by_name=True)
    elif init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    return model

def load_and_display(dataset : CutsDataset):
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        print(mask.shape)
        print(image.shape)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the image directory', default=f"{workspace_dir}/data/coco_formatted_cuts_split/test")
    parser.add_argument('--save_path', type=str, help='Path to put the outputted images and mask data', default=f"{workspace_dir}/data/predictions")
    parser.add_argument('--weights', type=str, help='Path to the model weights directory', default=f"{workspace_dir}/data/weights/mask_rcnn_coco.h5")
    args = parser.parse_args()

    predict(args.data_path, args.save_path, args.weights)

