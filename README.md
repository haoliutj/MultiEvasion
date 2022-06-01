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

### Train malware detectors and get evaluation results (e.g. train malconv)
```
python3 train_models.py --model_name=malconv --input_size=102400 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/inputsize_102400/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
```

##### Parameters
```
--model_name: name of the model, include "malconv", "fireeye" and "AvastNet"
--input_size: input size of models
--window_size: stride size 
--batch_size: batch size
--epochs: number of epochs to train models
--lr: learning rate
--num_workers: number of workers to load data
--log_file_path: file path to save the results
--checkpoint_dir: parent folder path to save model or load model
--train_label_path: csv file that includes file names and the corresponding labels of training data
--test_label_path: csv file that includes file names and the corresponding labels of testing data
--val_label_path: csv file that includes file names and the corresponding labels of validation data
--all_file_path: folder path that includes all files (pe programs), which include both benign and malcious pe programs
```

### Train image based malware detector, ResNet18
```
python3 train_image_malware_detector.py --epochs=50 --batch_size=128 --model_name=ResNet18 --lr=0.001 --image_resolution=320 --adjust_lr_flag=False --image_path=../data/all_image/ --model_save_path=../checkpoint/model_image_.pth --log_file_name=../result/model_training/log_image_.txt --train_label_table_path=../data/train_data_label.csv --test_label_table_path=../data/test_data_label.csv --val_label_table_path=../data/val_data_label.csv
```

##### Parameters
```
--epochs: number of epochs
--batch_size: batch size
--model_name: model name, default is ResNet18
--lr: learning rate
--image_resolution: the size of width (or height). height is equal to width here
--adjust_lr_flag: bool value, whether adjusting learning rate during training process
--image_path: folder path that includes all grayscale images of all benign and malicious pe programs
--train_label_path: csv file that includes file names and the corresponding labels of training data
--test_label_table_path: csv file that includes file names and the corresponding labels of testing data
--val_label_table_path: csv file that includes file names and the corresponding labels of validation data
```

### Evaluation of MultiEvasion against two malware detectors (MalConv and FireEyeNet) at the same time

##### FGSM as adversarial example algorithm, Content Shift and Slack as function manipulations:
```
python3 adv_attack_against_detectors.py --adversary=FGSM  --eps=0.4 --alpha=0.7 --iter_steps=20 --partial_dos=False --content_shift=True --slack=True --combine_w_slack=False --log_file_for_result=../result/same_input_format/fgsm_0.4/result_file.txt --preferable_extension_amount=0 --preferable_shift_amount=4096 --test_data_path=../data/all_file/ --test_label_path=../data/test_mal_label.csv --model_path_1=../checkpoint/malconv_model.pth --model_path_2=../checkpoint/fireeye_model.pth --use_cpu=1 --batch_size=1 --first_n_byte=102400 --window_size=500

```

##### Parameters
```
--adversary: name of adversary, include "FGSM", "FFGSM" and "PGD"
--eps: control the perturbation size, range [0,1]
--alpha: the maximum perturbation size of each iteration (for PGD)
--iter_steps: number of iteration steps to perform evasion attacks (for PGD)
--partial_dos: bool value, if True, select Partial DOS as function manipulation 
--content_shift: bool value, if True, select Content Shift as function manipulation
--slack: bool value, if True, select Slack as function manipulation
--combine_w_slack: bool value, if True, add Slack as an additional function manipulation
--log_file_result: file path the save the results
--preferable_extension_amount: the number of bytes to perform Extension function manipulation (should be multiple of 512)
--preferable_shit_amount: the number of bytes to perform Content Shift function manupulation (should be multiple of 512)
--test_data_path: the file path that includes all testing data (testing pe programs)
--test_label_path: csv file that includes file names and the corresponding labels of testing data
--model_path_1: the trained model path of model 1 (default is MalConv)
--model_path_1: the trained model path of model 2 (default is FireEyeNet)
--batch_size: batch size
--use_cpu: number of workers/cpu cores to load data
--first_n_byte: input size of models
--window_size: the stride size of models
```

##### How to select function manipulations, please follow the below parameter combinations to choose function manipulations

Partial DOS:
```
--partial_dos=True --content_shift=False --slack=False --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=0
```

Content Shift: 
```
--partial_dos=False --content_shift=True --slack=False --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=512
```

Slack: 
```
--partial_dos=False --content_shift=False --slack=True --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=0
```

Partial DOS + Content Shift: 
```
--partial_dos=True --content_shift=True --slack=False --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=512
```

Partial DOS + Slack:
```
--partial_dos=True --content_shift=False --slack=True --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=0
```

Content Shift + Slack: 
```
--partial_dos=False --content_shift=True --slack=True --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=512
```

Partial DOS + Content Shift + Slack:
```
--partial_dos=True --content_shift=True --slack=True --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=512
```

Full DOS: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=0
```

Extension: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=False --preferable_extension_amount=512 --preferable_shift_amount=0
```

Full DOS + Content Shift: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=False --preferable_extension_amount=0 --preferable_shift_amount=512
```

Full DOS + Slack: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=True --preferable_extension_amount=0 --preferable_shift_amount=0
```

Extension + Content Shift: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=False --preferable_extension_amount=512 --preferable_shift_amount=512
```

Extension + Slack: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=True --preferable_extension_amount=32768 --preferable_shift_amount=0
```

Full DOS + Content Shift + Slack: 
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=True --preferable_extension_amount=0 --preferable_shift_amount=32768
```

Extension + Content Shift + Slack:
```
--partial_dos=False --content_shift=False --slack=False --combine_w_slack=True --preferable_extension_amount=512 --preferable_shift_amount=512
```


### Evaluation of MultiEvasion against three malware detectors (MalConv, FireEyeNet and AvastNet) at the same time
##### FGSM as adversarial example algorithm, Content shift and slack as function manipulation:
```
python3 adv_attack_against_3detectors.py --adversary=FGSM  --eps=0.9 --alpha=0.7 --iter_steps=20 --partial_dos=False --content_shift=True --slack=True --combine_w_slack=False --log_file_for_result=../result/same_input_format_3detectors/fgsm_0.9/result_file.txt --preferable_extension_amount=0 --preferable_shift_amount=4096 --test_data_path=../data/all_file/ --test_label_path=../data/test_mal_label.csv --model_path_1=../checkpoint/malconv_model.pth --model_path_2=../checkpoint/fireeye_model.pth --model_path_3=../checkpoint/AvastNet_model.pth --use_cpu=1 --batch_size=1 --first_n_byte=102400 --window_size=500
```


##### Parameters
The parameters decriptions same as above.


### Evaluation of MultiEvasion against two malware detectors with different input formats(MalConv and ResNet18) at the same time
##### FGSM as adversarial example algorithm, Content shift and slack as function manipulation:
```
python3 adv_attack_against_detectors_different_input.py --adversary=FGSM  --eps=0.4 --alpha=0.7 --iter_steps=20 --partial_dos=False --content_shift=True --slack=True --combine_w_slack=False --log_file_for_result=../result/different_input/fgsm0.4/result_file.txt --preferable_extension_amount=0 --preferable_shift_amount=4096 --width=320 --height=320 --test_data_path=../data/all_file/ --test_label_path=../data/test_mal_label.csv --model_path_1=../checkpoint/malconv_model.pth --model_path_2=../checkpoint/ResNet18.pth --use_cpu=1 --batch_size=1 --first_n_byte=102400 --window_size=500

```

##### Parameters
The most parameters decriptions same as above.
```
--width: width value of image. Default is 320.
--height: height value of image. Default is 320.
```


## Citation
When reporting results that use the dataset or code in this repository, please cite:

Hao Liu, Wenhai Sun, Nai Niu, Boyang Wang, "AdvTraffic: Obfuscating Encrypted Traffic with
Adversarial Examples" (under review)

## Contacts
Hao Liu, liu3ho@mail.uc.edu

Boyang Wang, boyang.wang@uc.edu
