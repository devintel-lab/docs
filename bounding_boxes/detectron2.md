# Object detection using detectron2

A fully functioning version of detectron2 is available on einstein (the Lambda GPU workstation) in this directory: ```/data/code/obj_loc/detectron2```. 

This folder is a git repository which is also hosted on our [github](https://github.com/devintel-lab/detectron2).

## Activate virtual environment

To run the code in the copy of the repo on einstein, you'll first need to activate its Python virtual environment:

First, change your working directory to the repo:

```
$ cd /data/code/obj_loc/detectron2
```

Then activate the virtual environment:

```
$ source env/bin/activate
```

This will load a local python environment where all the dependencies required for running the code are installed and ready for use. This means when you run any script with ```python script.py```, it will call the python interpreter that's in the repo's ```env``` directory, rather than the global interpreter. This allows us to install the specific version of all the dependencies required for detectron2 (which may conflict with versions of those packages installed globally, i.e. pytorch)

To deactivate the environment, you can just run the deactivate command:

```
$ deactivate
```

# Adding a new dataset

You'll need to "register" a dataset with detectron2 to be able to train a model as well as run inference on a set of video frames.

There are two steps to this process. One is to produce a JSON file that documents all the image/bounding-box pairs, and the second step is to "register" that dataset with detectron2.

### 1) Create a Dataset JSON File

There are 2 types of datasets - one that's annotated with ground truth bounding boxes, and one that doesn't have any annotations and only lists a set of images. The first type is for training a model, while the second is for inference (i.e. after you already have a trained model, detect new boxes in images for which you have no ground truth labels).

Here's an example of a training JSON file with annotations:

```json
[
    {
        "file_name": "/path/to/image1.jpg",
        "image_id": 1,
        "width": 640,
        "height": 480,
        "annotations": [
            {
                "bbox": [
                223.0,
                148.00008,
                49.00032,
                105.99984
                ],
                "bbox_mode": 1,
                "category_id": 0
            },
            {
                "bbox": [
                85.99968000000001,
                12.000239999999991,
                77.00032,
                147.99984
                ],
                "bbox_mode": 1,
                "category_id": 4
            }
        ]
    },
    {
        "file_name": "/path/to/image2.jpg",
        "image_id": 2,
        "width": 640,
        "height": 480,
        "annotations": [
            {
                "bbox": [
                107.0,
                254.0,
                13.0,
                67.0
                ],
                "bbox_mode": 1,
                "category_id": 6
            }
        ]
    }
]
```

This example dataset has 2 images, where the first image has 2 ground truth boxes annotated, and the second image has only 1 bounding box annotated. The ```"category_id"``` key in each box annotation contains the class of each bounding box (i.e. the category of object, e.g. "bison" = 0, "aligator" = 1, etc...)

Here's an example of an inference dataset (i.e. no ground truth boxes) containing only 2 images:

```json
[
    {
        "file_name": "/path/to/image1.jpg",
        "image_id": 1,
        "width": 640,
        "height": 480,
        "annotations": []
    },
    {
        "file_name": "/path/to/image2.jpg",
        "image_id": 2,
        "width": 640,
        "height": 480,
        "annotations": []
    }
]
```

There's a helper utility script available if you want to generate an inference JSON file and all you have is a folder full of images you want to detect boxes for. This script, [make_inference_set.py](https://github.com/devintel-lab/detectron2/blob/main/datasets/make_inference_set.py), is in the ```datasets``` directory.

To run this script, you need to provide it with 3 argument:

```
$ python make_inference_set.py --image_dir /path/to/dir/with/images --dataset_name my_dataset --output_dir /path/to/output/dir
```

This script will read all the images in the directory specified by the ```--image_dir``` argument and output a file named ```my_dataset.json``` in the directory specified by the ```--output_dir``` argument


### 2) Register a Dataset with detectron2

This step is where you tell detectron2 how to load the data from the JSON file you made in step 1). To do this you must "register" the dataset.

You can see an example of a [dataset registration script](https://github.com/devintel-lab/detectron2/blob/main/datasets/home15.py) for experiment 15 in the dataset folder of the detectron2 repo.

This involves defining a function which returns a list of python dictionaries with all the dataset info. This list of dictionaries are what's encoded in the JSON file we made in step 1), and we can load and return this data structure as demonstrated in the dataset registration script mentioned previously:

```python
def home_15_train_dataset():
    path = osp.join(osp.dirname(__file__), "home15_train.json")
    with open(path, "r") as input:
        data = json.load(input)
        return data
```

After that function has been defined, we associate that function with a dataset name, and "register" it with detectron2:

```python
from detectron2.data import DatasetCatalog
DatasetCatalog.register("home15_train", home_15_train_dataset)
```

In the above example we've registered a dataset named "home15_train" and associated it with the function ```home_15_train_dataset``` which we defined above.

You'll also want to set a few metadata variables, like what each of the numeric "category_id" values refer to (in order from 0 to N):

```python
home15_classes = ["bison", "alligator", "drop", 
                 "kettle", "koala", "lemon", "mango", 
                 "moose", "pot", "seal", "pot_yellow", 
                 "pot_black"]

from detectron2.data import MetadataCatalog

MetadataCatalog.get("home15_train").set(thing_classes = home15_classes)
```

As well as the method of performing evaluation (i.e. which evaluator to use when running inference):

```python
from detectron2.evaluation import COCOEvaluator

MetadataCatalog.get("home15_train").set(evaluator_type="coco")

```


# Training

To train a model, you'll need to first set up a config file that outlines all the parameters for training/testing your model. You can find an example config for the ```home15_train``` and ```home15_test``` dataset [here](https://github.com/devintel-lab/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_HOME15.yaml). 

The ```home15_test``` dataset is just a set of 10% of left-out annotated frames which we can be used for testing (i.e. computing precision/recall performance).

After you have this config file, you will pass it to the [```train_net.py```](https://github.com/devintel-lab/detectron2/blob/main/tools/train_net.py):

```
$ python train_net.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_HOME15.yaml
```

This will begin training and output checkpointed model weights to the directory specified in the ```OUTPUT_DIR``` parameter in the config file.

For this example, before we start training, we'll also have to make sure to import the dataset registration script we defined earlier so that detectron2 knows about our datasets. To do this, add the registration script as an import to the top of the [```train_net.py```](https://github.com/devintel-lab/detectron2/blob/main/tools/train_net.py) script:


```python
sys.path.append("../datasets")
import home15
```

# Inference

Inference is the step where you perform object detection on new data after you have a trained model.

Assuming you generated an inference dataset JSON and registered that dataset with detectron2 (following the previously outlined steps for adding a new dataset), you'll need to make another config file for this inference task.

We have an example inference config you can check out [here](https://github.com/devintel-lab/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_HOME15_RANDOMSAMPLE_INFERENCE.yaml). This config uses the ```home15_randomsample_inference``` dataset, which was generated using ```make_inference_set.py```, and refers to a set of ~2200 randomly rample images from experiment 15. 

You'll notice that in this config file the ```WEIGHTS``` parameter is a path pointing to the latest checkpointed weights from a previously trained model: ```"/data/code/obj_loc/detectron2/tools/output/faster_rcnn_R_50_FPN_3x_HOME15.yaml/model_0074999.pth"```. This is the set of weights learned by a model trained using the ```faster_rcnn_R_50_FPN_3x_HOME15.yaml``` config file, but applied for detecting boxes in the  ```home15_randomsample_inference``` dataset.

To run inference you'll again be calling the [```train_net.py```](https://github.com/devintel-lab/detectron2/blob/main/tools/train_net.py) script:

```
$ python train_net.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_HOME15_RANDOMSAMPLE_INFERENCE.yaml --eval-only
```

Notice the ```--eval-only``` flag we also pass in here. This tells the detectron2 system that we're not doing any training here, we're just evaluating.

This step will produce a set of results in a folder named ```inference``` in the ```OUTPUT_DIR``` specified in the config. The bounding box detections will be in a file called ```coco_instances_results.json```, and a manifest of all the images and their ID's will be in a file called ```{dataset_name}_coco_format.json```. The ```coco_instances_results.json``` file will tell you that an image with id X has certain boxes detected, and the ```{dataset_name}_coco_format.json``` file will tell you which image ID corresponds to which image file.

To visualize these detections we have a few helper utility scripts that can generate a collage of all the the detections in each category.


# Checking Detections

To check the quality of detections we have a few helper utility scripts. The first, [```crop_items.py```](https://github.com/devintel-lab/detectron2/blob/main/tools/crop_items.py) will crop images of each category following the bounding box detections, and the second, [```collage.py```](https://github.com/devintel-lab/detectron2/blob/main/tools/collage.py) will stitch all these cropped boxes into a collage so we can view all of them together.

### crop_items.py

```
$ python crop_items.py --bbox_file /path/to/{dataset_name}_coco_format.json --coco_instances_file /path/to/coco_instances_results.json --output_dir /path/to/output
```

This will generate 1 directory for each category of object and fill these directories with crops of the detected instances of these objects in the dataset.

### collage.py

The ```collage.py``` script will take a folder filled with cropped instances of a single category of object and stitch these images together into a collage. 

```
$ python collage.py --box_image_dir /path/to/cropped/boxes/categoryX --out_dir /path/to/output
```

the ```--box_image_dir``` argument is the path to one of the directories of object crops created by the ```crop_items.py``` script (corresponding to crops of a single category), and the ```--out_dir``` argument is the directory where the script will save a JPG file containing a collage of all these crops.



