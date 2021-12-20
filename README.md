# Fishial.ai

This project includes training and validation scripts for the fish segmentation and classification model.

[Demo web aplication](https://portal.fishial.ai/search)

Project website: www.fishial.ai

## Installation

Install the dependencies.

```sh
$ pip3 install -r requirements.txt
```

## Getting Started
* [InferenceTest.ipynb](helper/classification/InferenceTest.ipynb) Is the easiest way to start. It shows an example of using a model train Mask RCNN and classification model ResNet18. It includes code to run fish instance segmentation on your images and classification they.

* [UniformDataset.ipynb](helper/classification/UniformDataset.ipynb) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [EvalNoteFC.ipynb](helper/classification/EvalNoteFC.ipynb) is the script validate trained FC model on specific Dataset (Plot confusion matrix, recall, precision, etc. )

* [EvalNoteEmbed.ipynb](helper/classification/EvalNoteEmbed.ipynb)  is the script validate trained Embeding Network on specific Dataset (Plot confusion matrix, recall, precision, etc. )

* [VectorCreator.ipynb](helper/classification/VectorCreator.ipynb) the script creates a dictionary with a list of attachments corresponding to each class.

* [CopyPasteDebug.ipynb](helper/segmentation/CopyPasteDebug.ipynb) is the script shows how to implement CopyPaste debug realesed by [conradry](https://github.com/conradry/copy-paste-aug "conradry")

* [UniformDataset.ipynb](helper/classification/UniformDataset.ipynb) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script performs training automatically with different parameters the selected model using the cross entropy loss function. The checkpoint with the best performance on the validation dataset is saved to the output folder.

* [auto_train_triplet.py](train_scripts/classification/auto_train_triplet.py) is the script performs training automatically with different the selected model using the (Triplet Quadruplet) loss function. The checkpoint with the best performance on the validation dataset according k-metric is saved to the output folder.

* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [train.py](train_scripts/segmentation/train.py) is the basic set up script to train segmentation model using Detectrin2 API 

* [train_copy_paste.py](train_scripts/segmentation/train_copy_paste.py) is the basic set up script to train segmentation model using Detectrin2 API with Copy Paste Augumentation.

* ([model.py](module/classification_package/src/model.py), [utils.py](module/classification_package/src/utils.py)): These files contain the main classification pipline implementation.

## Models

| Model | link  |
| ------------- | ------------- |
| MaskRCNN Fish Segmentation  | [link](https://storage.googleapis.com/fishial-ml-resources/detectron2_new_version.pth) |
| ResNet18 Fish Classification  | [link](https://storage.googleapis.com/fishial-ml-resources/final_cross_cross_entropy_0.9923599320882852_258571.0.ckpt) |

## [Train results](train_scripts/README.md)


Segmentation model has validated by mAP metric.

**MaskRCNN **

| AP | AP50  | AP75 | APs | APm | APl | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 75.085  | 91.120 | 85.695 | 29.064 | 54.678 | 80.310 |


**Classification model**

**Cohen kappa:** 0.83

Confusion matrix:
![Confusion matrix](imgs/image5.png "Confusion matrix")


<p float="left">
  <img src="https://fishial.ai/static/fishial_logo-2c651a547f55002df228d91f57178377.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2020/08/68e6fe03-e654-4d15-9161-98715ff1f393.png" height="40" /> 
  <img src="https://wp.fishial.ai/wp-content/uploads/2021/01/WYE-Foundation-Full-Color.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2019/08/dotcom-standard.png" height="40" />
</p>


## License

[MIT](https://choosealicense.com/licenses/mit/)

