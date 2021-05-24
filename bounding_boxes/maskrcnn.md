# Object Detection using maskrcnn-benchmark


A fully functioning version of Mask-RCNN is available on Salk in this directory: ```/data/aamatuni/code/obj_loc```. 

## Directory structure

 - ```apex```
    - Dependency required for compiling maskrcnn
 - ```cityscapesScripts```
    - Dependency required for compiling maskrcnn
 - ```cocoapi```
    - the COCO dependency
 - ```maskrcnn-benchmark```
    - Where all the object detection code is
 - ```venv```
    - The virtual environment. This needs to be loaded before you can start using the system


## Add a New Dataset

These are the steps for training mask-rcnn on a new dataset.

1. Annotate ground truth bounding boxes for the dataset using the MATLAB based [annotation tool](https://github.com/devintel-lab/yolo_processing/blob/master/coding_tools/OBJECT_BOX_GUI_HOME/label_toys.m)
    - after these annotations have been completed, run the [```create_training_data_from_annotations.m```](https://github.com/devintel-lab/yolo_processing/blob/master/prepare_training_data/create_training_data_from_annotations.m) function on the directory where the annotation software saved outputs (which should be .mat files). This will generate a directory that has the following structure:
        - ```JPEGImages``` 
            - directory containing all the images that were annotated
        - ```labels```
            - directory containing a record of all the ground truth bounding box annotations
        - ```training.txt```
            - a manifest file containing a list of all the annotated images
1. Create a dataset json file using [```home2coco.py```](https://github.com/devintel-lab/home2coco/blob/master/home2coco.py)
    - required arguments: 
        - ```--input_dir```
            - path to the folder that contains ```JPEGImages```, ```labels```, and ```training.txt```
        - ```--out_dir```
            - path where you want to save the dataset json file that's generated by the script
        - ```--exp```
            - the experiment number, e.g. '15'
        - There are more arguments you can set which are outlined in the argument parsing [section](https://github.com/devintel-lab/home2coco/blob/master/home2coco.py#L42) of the script, but the only required ones for this step are those mentioned above. Some of these will become necessary when building the inference dataset json (as opposed to the training dataset)
    - The output from this script is a json file that contains a record of all the images and associated ground truth bounding box annotations. 
    - You need to add this json file to the ```data``` directory of the ```maskrcnn-benchmark``` repository located on salk (*NOTE*: this folder is not present on the github copy of the repo, only the Salk version, which is located at ```/data/aamatuni/code/obj_loc```)
1. Add paths for image directory and dataset json to [```paths_catalog.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/config/paths_catalog.py)
    - This registers an entry for the dataset annotation files and image directory paths for maskrcnn to use.
    - Here is an example entry for the training dataset for exp15
        ```python
        "home_15_train_multipot": {
                    "img_dir": "home/train15",
                    "ann_file": "home/annotations/train15_multipot.json"
                },
        ```
        For this entry the directory with all the training images is ```data/home/train15```, while the dataset json is a file called ```train15_multipot.json``` and located in ```data/home/annotations```. Note that all these paths are relative to the ```data``` directory of the maskrcnn-benchmark repo. The name of this entry is ```"home_15_train_multipot"```. You should give your dataset a unique name that's easily interpretable. 
1. Create a new dataset definition in the [```datasets```](https://github.com/devintel-lab/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/data/datasets) module of the ```maskrcnn-benchmark``` repo
    - Each dataset is defined here within its own python file. You'll notice that there's already entries here for the [```home```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/home.py) dataset and the [```toyroom```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/toyroom.py) datasets. Use these as a point of reference when defining your new dataset.
    - Add the name of this new dataset (i.e. the name of the class you defined in the python file you just created, e.g. ```HOMEDataset``` or  ```ToyroomDataset```), to this [```__init__.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/__init__.py) file.
    - Add this new dataset as one of the conditions in the evaluation [```__init__.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/datasets/evaluation/__init__.py#L21) file.
        - you'll need to add it so that this new dataset class is evaluated as if it were a ```COCODataset```. You'll nontice the ```HOMEDataset``` and ```ToyroomDataset``` classes are also handled as if they were a ```COCODataset```. Your new dataset should be appended at the end of this 'if' statement so that it's handled the same as these. For example, if your dataset class is called ```PBJDataset```, then you should append 
            ```
            or isinstance(dataset, datasets.PBJDataset)
            ```
            to the end of this ```if``` statement before the final "```:```"
1. Create a training/inference config file
    - maskrcnn expects a config file to be passed in as a command line argument when running training as well as inference. These config files (in YAML format) are located in the [```configs```](https://github.com/devintel-lab/maskrcnn-benchmark/tree/master/configs) directory of the repo. 
    - You can use the [HOME multipot training config](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/configs/e2e_faster_rcnn_R_50_FPN_1x_HOME_multipot.yaml) file as as a reference point to make your own config file. You'll need to change a couple values here:
        - ```NUM_CLASSES``` - should be the number of distinct object classes in your dataset + 1. So if you have 10 objects, ```NUM_CLASSES``` should be 11. 
        - ```DATASETS``` - should be the names of the entries you added to the ```paths_catalog.py``` file in step 3 for your new dataset. If this was the exp15 multipot training dataset from the example in step 3, then the name you would add here would be ```"home_15_train_multipot"```
        - ```OUTPUT_DIR``` - this is where maskrcnn will save checkpoints and detections


## Run Training

1. Start the virtual environment in the root of the ```obj_loc``` directory on salk. Assuming your current working directory is ```/data/aamatuni/code/obj_loc```, run the following command:
    - ```
        $ source ./venv/bin/activate
        ```
        - this will load all the dependencies necessary for the code to run
1. The [```train_net.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/tools/train_net.py) script is located in the [```tools```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/tools) directory. The script expects a single command line argument:
    - ```--config-file```
        - this is the path to the YAML config file you created in step 5 of the "Add a New Dataset" section. 


## Run Inference

In order to run object detection inference on un-annotated frames, you'll need to build a new dataset json file and register it with the maskrcnn system. Note, this is not about defining a new dataset _type_ as we did in step 4, but rather creating a new instance of input data for an already existing type of dataset (in the form of a new json file and new image directory) , i.e. steps 2 and 3.

1. Crawl multiwork to copy over all the frames associated with a dataset. This will build a folder filled with all the frames and helper files we'll need to compute box detections
    - to do this, use the [```crawl.py```](https://github.com/devintel-lab/home2coco/blob/master/crawl.py) script that's located in the [```home2coco```](https://github.com/devintel-lab/home2coco) repo.
        - required arguments:
            - ```--exp_dir```
                - the path to the experiment on multiwork, e.g. ```/marr/multiwork/experiment_15```
            - ```--out_dir```
                - this is where the script will output all image frames and associated files
            - ```--exp```
                - this is the experiment number
    - The output of this script are 2 directories: ```JPEGImages``` and ```labels```, and 1 ```training.txt``` file. You'll recognize that these were the same type of outputs from the [```create_training_data_from_annotations.m```](https://github.com/devintel-lab/yolo_processing/blob/master/prepare_training_data/create_training_data_from_annotations.m) function in step 1 of Add a New Dataset, which we then used as inputs to [```home2coco.py```](https://github.com/devintel-lab/home2coco/blob/master/home2coco.py) in step 2.
1. Generate a new inference dataset using [```home2coco.py```](https://github.com/devintel-lab/home2coco/blob/master/home2coco.py)
    - required arguments: 
        - ```--input_dir```
            - path to the folder that contains ```JPEGImages```, ```labels```, and ```training.txt```
        - ```--out_dir```
            - path where you want to save the dataset json file that's generated by the script
        - ```--exp```
            - the experiment number, e.g. '15'
        - ```--infer_set```
            - this flag tells home2coco that this is an inference set and there are no ground truth labels associated to the images. The flag should just be appended to the command without a corresponding argument, e.g: 
                ```
                python home2coco.py --input_dir input --out_dir output --exp 15 --infer_set
                ```
    - optional arguments
        - ```--samp_percent```
            - subsample a random percent of the full set of files
    - the output of this script will be a json file defining this new inference dataset.
1. Create a new dataset entry in the [```paths_catalog.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/config/paths_catalog.py) file for this inference set. 
    - You'll need to copy over the ```JPEGImages``` folder to the ```data``` directory of maskrcnn-benchmark repo so that maskrcnn can load these images to run box detection on them. Give this new folder an informative name. 
    - Here is an example entry for the inference dataset for exp15
        ```python
        "home_15_infer_multipot": {
            "img_dir": "home/infer15",
            "ann_file": "home/annotations/test15_multipot_infer.json"
        },
        ```
        - the ```infer15``` folder is a renamed version of ```JPEGImages``` which contains all the images that were collected from crawling in step 1.
        - the ```test15_multipot_infer.json``` is the json output from the [```home2coco.py```](https://github.com/devintel-lab/home2coco/blob/master/home2coco.py) script in the previous step
1. Create a config file for inference
    - We need to create a new config file for this inference set. You can use the [```e2e_faster_rcnn_R_50_FPN_1x_HOME_infer_multipot.yaml```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/configs/e2e_faster_rcnn_R_50_FPN_1x_HOME_infer_multipot.yaml) config as a reference point for defining your config file. You'll need to change the following values:
        - ```DATASETS```
            - this is the name of the inference dataset you created in the previous step, e.g. ```"home_15_infer_multipot"```
        - ```OUTPUT_DIR```
            - a unique name for where to save model checkpoints and detection results
1. Run inference
    - to run inference you'll need to use the [```test_net.py```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/tools/test_net.py) function located in the [```tools```](https://github.com/devintel-lab/maskrcnn-benchmark/blob/master/tools) directory of maskrcnn. 
    - call the ```test_net.py``` script with the ```--config-file``` command line argument pointing to the config file you created in the previous step
    - detection results will be saved in a directory called ```inference``` within the ```OUTPUT_DIR``` specified in the config file