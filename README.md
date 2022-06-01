# MultiEvasion

This repository contains the implementation of our proposed evasion attacks "MultiEvasion" that can evade multiple malware detectors. We leveraged different adversarial example generation methods, like FGSM, FFGSM, PGD, which orginally introduced in image domain, we customized them to apply on obfuscating malware programs.

** The data set and code for research purpose only**

## Environment
To running this repositery, we recommed you to install the environment with advmal.txt file:
```conda env create --file advmal.txt```


## Content

This repository contains separate directories, including "attacks", "src". A brief description of the contents of these directories is below.  More detailed usage instructions are found in the following contents.

The ```attacks``` directory contains the scripts for generating adversarial malware with different adversarial algorithms and function manipulations. 

The ```src``` directory contains the scipts for trianing and testing deep learning based malware detectors, evaluating on evading multiple malware detectors (with same/different input format (e.g., raw bytes or images)) at the same time.


## Datasets

We evaluated our method over two datasets. One is public dataset, named phd dataset, contains 977 benign PE programs and 2597 malicious PE programs. You can access the dataset through "https://github.com/tgrzinic/phd-dataset". The other one is collected by us, named ME dataset, contains 1000 benign PE programs across multiple Windows versions (e.g., Windows XP, Vista, 8 and 10), and 1000 malicious PE programs downloaded from VirusShare. Please contact us for the link to download the ME dataset if you need to reproduce our results or conduct related research. 



## Usage
Please see the details of parameters in each script.

#### Train malware detectors and get evaluation results (saved at "log_file_path") (e.g. train malconv)
```python3 train_models.py --model_name=malconv --input_size=102400 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/inputsize_102400/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/```

```
model_name: name of the model, include "malconv", "fireeye" and "AvastNet"
input_size: input size of models
window_size: stride size 
batch_size: batch size
epochs: number of epochs to train models
lr: learning rate
num_workers: number of workers to load data
log_file_path: file path to save the results
checkpoint_dir: parent folder path to save model or load model
train_label_path: csv file that includes file names and the corresponding labels of training data
test_label_path: csv file that includes file names and the corresponding labels of testing data
val_label_path: csv file that includes file names and the corresponding labels of validation data
all_file_path: folder path that includes all files (pe programs), which include both benign and malcious pe programs
```

#### Train image based malware detector, ResNet18
```python train_image_malware_detector.py --epochs=50 --batch_size=128 --model_name=ResNet18 --lr=0.001 --image_resolution=320 --adjust_lr_flag=False --image_path=../data/all_image/ --model_save_path=../checkpoint/model_image_.pth --log_file_name=../result/model_training/log_image_.txt --train_label_table_path=../data/train_data_label.csv --test_label_table_path=../data/test_data_label.csv --val_label_table_path=../data/val_data_label.csv```

#### Evaluation






## Citation
When reporting results that use the dataset or code in this repository, please cite:

Hao Liu, Wenhai Sun, Nai Niu, Boyang Wang, "AdvTraffic: Obfuscating Encrypted Traffic with
Adversarial Examples" (under review)

## Contacts
Hao Liu, liu3ho@mail.uc.edu

Boyang Wang, boyang.wang@uc.edu
