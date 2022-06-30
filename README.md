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
* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script performs training automatically with different parameters the selected model using the cross entropy loss function. The checkpoint with the best performance on the validation dataset is saved to the output folder.

* [auto_train_triplet.py](train_scripts/classification/auto_train_triplet.py) is the script performs training automatically with different the selected model using the (Triplet Quadruplet) loss function. The checkpoint with the best performance on the validation dataset according k-metric is saved to the output folder.

* [auto_train_cross.py](train_scripts/classification/auto_train_cross.py) is the script cut your dataset to specific maximum and minimum count images per class to the arbitrary way.

* [train.py](train_scripts/segmentation/train.py) is the basic set up script to train segmentation model using Detectrin2 API 

* [train_copy_paste.py](train_scripts/segmentation/train_copy_paste.py) is the basic set up script to train segmentation model using Detectrin2 API with Copy Paste Augumentation.

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


## [Train results](train_scripts/README.md)


Segmentation model has validated by mAP metric.

**MaskRCNN **

| AP | AP50  | AP75 | APs | APm | APl | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 82.504  | 96.742 | 94.727 | 13.283 | 58.029 | 84.540 |


**Classification model**

Names of fish classes in tensor
```json
{
"0": "Lepomis macrochirus", "1": "Pomatomus saltatrix", "2": "Pogonias cromis", "3": "Esox lucius", "4": "Amia calva", "5": "Micropogonias undulatus", "6": "Micropterus salmoides", "7": "Lachnolaimus maximus", "8": "Scomberomorus cavalla", "9": "Coryphaena hippurus", 
"10": "Sciaenops ocellatus", "11": "Cyprinus carpio", "12": "Caranx hippos", "13": "Lobotes surinamensis", "14": "Lutjanus analis", "15": "Scomberomorus maculatus", "16": "Oncorhynchus clarkii", "17": "Lepomis gulosus", "18": "Perca fluviatilis", "19": "Pomoxis nigromaculatus", 
"20": "Lutjanus argentimaculatus", "21": "Salvelinus fontinalis", "22": "Oncorhynchus tshawytscha", "23": "Ambloplites rupestris", "24": "Cynoscion nebulosus", "25": "Pomoxis annularis", "26": "Morone saxatilis", "27": "Mycteroperca bonaci", "28": "Seriola dumerili", "29": "Morone chrysops", 
"30": "Cichla ocellaris", "31": "Esox niger", "32": "Lepomis cyanellus", "33": "Esox masquinongy", "34": "Sphyraena barracuda", "35": "Ictalurus punctatus", "36": "Lutjanus synagris", "37": "Lutjanus griseus", "38": "Dicentrarchus labrax", "39": "Centropristis striata", 
"40": "Lepomis gibbosus", "41": "Salmo trutta", "42": "Acanthocybium solandri", "43": "Centropomus undecimalis", "44": "Paralichthys dentatus", "45": "Ameiurus nebulosus", "46": "Salvelinus namaycush", "47": "Pylodictis olivaris", "48": "Archosargus probatocephalus", "49": "Ocyurus chrysurus", 
"50": "Sander vitreus", "51": "Lepisosteus osseus", "52": "Platycephalus fuscus", "53": "Lepomis auritus", "54": "Micropterus punctulatus", "55": "Ameiurus melas", "56": "Oncorhynchus mykiss", "57": "Ictalurus furcatus", "58": "Rutilus rutilus", "59": "Lepomis microlophus", 
"60": "Micropterus dolomieu", "61": "Abramis brama", "62": "Epinephelus morio", "63": "Paralichthys lethostigma", "64": "Aplodinotus grunniens", "65": "Morone americana", "66": "Tilapia sparrmanii", "67": "Perca flavescens", "68": "Balistes capriscus", "69": "Lutjanus campechanus", 
"70": "Ctenopharyngodon idella", "71": "Scomber scombrus", "72": "Caranx ignobilis", "73": "Carcharhinus limbatus", "74": "Mycteroperca microlepis", "75": "Belone belone", "76": "Oncorhynchus kisutch", "77": "Thunnus albacares", "78": "Ariopsis felis", "79": "Bagre marinus", 
"80": "Cyprinus carpio carpio", "81": "Mycteroperca venenosa", "82": "Girella elevata", "83": "Mustelus canis", "84": "Lutjanus cyanopterus", "85": "Rhincodon typus", "86": "Ameiurus catus", "87": "Oncorhynchus nerka", "88": "Rachycentron canadum", "89": "Sarda sarda", 
"90": "Megalops atlanticus", "91": "Lates calcarifer", "92": "Thunnus atlanticus", "93": "Xiphias gladius", "94": "Euthynnus alletteratus", "95": "Pterois volitans", "96": "Carassius auratus", "97": "Cyprinus rubrofuscus"}
```

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

