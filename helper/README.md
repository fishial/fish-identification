# Fishial.ai

This folder contains working scripts for data processing.

### Description
* [InferenceTest.ipynb](classification/InferenceTest.ipynb) Is the easiest way to start. It shows an example of using a model train Mask RCNN and classification model ResNet18. It includes code to run fish instance segmentation on your images and classification they.

* [UniformDataset.ipynb](classification/UniformDataset.ipynb) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [EvalNoteFC.ipynb](classification/EvalNoteFC.ipynb) is the script validate trained FC model on specific Dataset (Plot confusion matrix, recall, precision, etc. )

* [EvalNoteEmbed.ipynb](classification/EvalNoteEmbed.ipynb)  is the script validate trained Embeding Network on specific Dataset (Plot confusion matrix, recall, precision, etc. )

* [VectorCreator.ipynb](classification/VectorCreator.ipynb) the script creates a dictionary with a list of attachments corresponding to each class.

* [CopyPasteDebug.ipynb](segmentation/CopyPasteDebug.ipynb) is the script shows how to implement CopyPaste debug realesed by [conradry](https://github.com/conradry/copy-paste-aug "conradry")

* [UniformDataset.ipynb](classification/UniformDataset.ipynb) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.


### COCOViewer

* [cocoviewer.py](cocoViewer/cocoviewer.py) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.
 This is a little bit changed version [this](https://github.com/trsvchn/coco-viewer) repository, which could load images from **coco_url**.  


### Generate dataBase for Classification model
* [predictDataEmbeddingClassificationModel.py](classification/predictDataEmbeddingClassificationModel.ipynb) - this notebook allows you to generate data for the embedding classification model.


### Voxel51

To register a model in the dataset in the voxel 1 application
necessary:
1. download all images to a specific folder to do this, run the following script:
* [downloader_coco_imgs.py](segmentation/downloader_coco_imgs.py) this script downloads all available images by **coco_url** tag in several threads to speed up the work.

to do this, you must specify two arguments: the path to the export file and the folder where to download the images.
example: 
```
python Object-Detection-Model/helper/segmentation/downloader_coco_imgs.py -c '/home/fishial/Fishial/dataset/export/03_export_Verified_ALL.json.json' -i '/home/fishial/Fishial/dataset/fishial_collection-test'
```

2. [converterCocoToVoxel.py](segmentation/converterCocoToVoxel.py) Run the script that converts the annotation file to voxel format and saves it.
```
python converterCocoToVoxel.py -c '/home/fishial/Fishial/dataset/export/03_export_Verified_ALL.json.json' -i '/home/fishial/Fishial/dataset/fishial_collection/data' -ds 'export-fishial-november-test'
```

#### Usage

```bash
python cocoviewer.py -a path_to_coco_file
```

### Сhecking masks

* [verifier_coco.py](cleaner/verifier_coco.py) This script checks the coco file for errors, such as: missing keys, incorrect key values, etc. All errors are saved in a json file and have the following structure:
```
{
    "3262841": {                            # image_id
        "image_warn": ["...", "...", ...],  # list of warnings in images key
        "annotations_warn": [
            {
                "2341": {                   # annotation id
                    ["...","...","..."]     # list of warnings in images key
                }
            }
        ]
    }
}

```

#### Usage

```bash
python verifier_coco.py -a path_to_coco_file
```

* [checking_masks.py](cleaner/checking_masks.py) This script, using a pre-trained neural network, checks whether the masks in the coco file are correct, as well as the script checks the number of segmentation points and the size of the masks relative to the whole image. After that, the cut out masks are saved in a separate folder for manual removal of false negative masks.

```bash
python checking_masks.py -a path_to_coco_file -sf path_to_dst_folder_with_masks --data_path path_to_folder_with_imgs (optional if not, script use a coco_url) 
```

* [generator.py](cleaner/generator.py) after a manual check, the remaining masks can be written to a json file for further processing.

Json has next structure:

```
{
    "3262841":                            # image_id
        [0001, 0022, 02174, ...]          # list of masks id
}

```

----

<p float="left">
  <img src="https://fishial.ai/static/fishial_logo-2c651a547f55002df228d91f57178377.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2020/08/68e6fe03-e654-4d15-9161-98715ff1f393.png" height="40" /> 
  <img src="https://wp.fishial.ai/wp-content/uploads/2021/01/WYE-Foundation-Full-Color.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2019/08/dotcom-standard.png" height="40" />
</p>


## License

[MIT](https://choosealicense.com/licenses/mit/)

