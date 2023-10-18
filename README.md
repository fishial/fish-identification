
# Fishial.ai

This project includes training and validation scripts for the fish segmentation and classification model.

[Demo web aplication](https://portal.fishial.ai/search/by-fishial-recognition)

Project website: www.fishial.ai

## Installation

Install the dependencies.

```sh
$ pip3 install -r requirements.txt
```

## Getting Started

* [Runner.ipynb](Runner.ipynb) This jupyter notebook allows you to run segmentation and classification neural networks on Google Cloud or your computer, after downloading the files from the links below.

* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script performs training automatically with different parameters the selected model using the cross entropy loss function. The checkpoint with the best performance on the validation dataset is saved to the output folder.

* [auto_train_triplet.py](train_scripts/classification/auto_train_triplet.py) is the script performs training automatically with different the selected model using the (Triplet Quadruplet) loss function. The checkpoint with the best performance on the validation dataset according k-metric is saved to the output folder.

* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [train.py](train_scripts/segmentation/train.py) is the basic set up script to train segmentation model using Detectrin2 API 

* [train_copy_paste.py](train_scripts/segmentation/train_copy_paste.py) is the basic set up script to train segmentation model using Detectrin2 API with Copy Paste Augumentation.

* [ExportModelToTorchscript.ipynb](helper/ExportModelToTorchscript.ipynb) This jupyter notebook allows convert Classification pytorch model to TorchScript format, and Detectron2 to Torchscript model.

* ([model.py](module/classification_package/src/model.py), [utils.py](module/classification_package/src/utils.py)): These files contain the main classification pipline implementation.

* [CreateDataBaseTensor.py](helper/classification/CreateDataBaseTensor.py): This script is designed to get the attachment tensor for a trained neural network, the resulting tensor has dimensions {number of classes * maximum number of images for one class * embedding dimension} for classes in which the number of attachments is less than in the maximum class, a tensor with a value of 100 is added in order not to affect for inference

* [CreateDatasetAndTrain.py](helper/classification/CreateDatasetAndTrain.ipynb): This script allows you to create a training and test data set from the exported fishial coco file and train the neural network

## Models

| Model | link  |
| ------------- | ------------- |
| MaskRCNN Fish Segmentation (Update 29.06.2022)  | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/model_0259999.pth) |
| ResNet18 Fish Classification Cross Entropy V1.0 | [link](https://storage.googleapis.com/fishial-ml-resources/final_cross_cross_entropy_0.9923599320882852_258571.0.ckpt) |
| ResNet18 Binary Classification  | [link](https://storage.cloud.google.com/fishial-ml-resources/binary_class.ckpt) |
| ResNet18 Fish Classification Embedding 256 V2.0  | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/full_256.ckpt) |
| ResNet18 DataBase Tensor  | [link](https://storage.googleapis.com/fishial-ml-resources/models_29.06.2022/train%2Btest_embedding.pt) |
| ResNet18 v4 model pack 184 classes | [link](https://storage.googleapis.com/fishial-ml-resources/classification_v5.zip) |
| MaskRCNN Fish Segmentation (Update 15.11.2022) (latest)  | [link](https://storage.googleapis.com/fishial-ml-resources/model_15_11_2022.pth) |
| ResNet18 v5 model pack 184 classes | [link](https://storage.googleapis.com/fishial-ml-resources/classification_22_12.zip) |
| ResNet18 v6 model pack 289 classes (latest) | [link](https://storage.googleapis.com/fishial-ml-resources/classification_fishial_30_06_2023.zip) |
**in the folder with the classification model ([V6](https://storage.googleapis.com/fishial-ml-resources/classification_fishial_30_06_2023.zip)) you will find an example of a python script (runner.py) how to run the model on your images**

## [Train results](train_scripts/README.md)


Segmentation model has validated by mAP metric.

**MaskRCNN **

| AP | AP50  | AP75 | APs | APm | APl | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 82.504  | 96.742 | 94.727 | 13.283 | 58.029 | 84.540 |


**Classification model**

Json file with the names of fish classes that the latest model recognizes can be found here: ([labels](labels.json)) 


<p float="left">
  <img src="https://fishial.ai/static/fishial_logo-2c651a547f55002df228d91f57178377.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2020/08/68e6fe03-e654-4d15-9161-98715ff1f393.png" height="40" /> 
  <img src="https://wp.fishial.ai/wp-content/uploads/2021/01/WYE-Foundation-Full-Color.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2019/08/dotcom-standard.png" height="40" />
</p>


## License

[MIT](https://choosealicense.com/licenses/mit/)

