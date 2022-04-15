##***************** our dataset ********************
##-------------> input_size=204800 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=204800 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/inputsize_204800/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/

### train malconv
#python train_models.py --model_name=malconv --input_size=204800 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/inputsize_204800/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=204800 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/inputsize_204800/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/


###-------------> input_size=409600 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=409600 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/inputsize_409600/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=409600 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/inputsize_409600/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=409600 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/inputsize_409600/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/


##-------------> input_size=1024000 <-------------
## train Avastnet
python train_models.py --model_name=AvastNet --input_size=1024000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_1024000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/

## train malconv
python train_models.py --model_name=malconv --input_size=1024000 --window_size=500 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_1024000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/

## train fireeye
python train_models.py --model_name=fireeye --input_size=1024000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_1024000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/


##-------------> input_size=102400 <-------------
## train Avastnet
python train_models.py --model_name=AvastNet --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/inputsize_102400/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/

## train malconv
python train_models.py --model_name=malconv --input_size=102400 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/inputsize_102400/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/

## train fireeye
python train_models.py --model_name=fireeye --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/inputsize_102400/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/


###-------------> input_size=2048000 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=2048000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_2048000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_2048000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=2048000 --window_size=500 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_2048000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_2048000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=2048000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/inputsize_2048000/train_log.txt --checkpoint_dir=../checkpoint/inputsize_2048000/ --train_label_path=../data/train_data_label.csv --test_label_path=../data/test_data_label.csv --val_label_path=../data/val_data_label.csv --all_file_path=../data/all_file/



###***************** phd dataset ********************
###-------------> input_size=102400 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_102400/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=102400 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_102400/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_102400/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
#
###-------------> input_size=204800 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=204800 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_204800/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=204800 --window_size=500 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_204800/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=204800 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_204800/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_204800/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
#
#
#
###-------------> input_size=409600 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=409600 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_409600/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=409600 --window_size=500 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_409600/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=409600 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_409600/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_409600/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
#
###-------------> input_size=1024000 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=1024000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_1024000/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=1024000 --window_size=500 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_1024000/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=1024000 --window_size=512 --batch_size=16 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset/inputsize_1024000/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset/inputsize_1024000/ --train_label_path=../data/train_data_label_phd.csv --test_label_path=../data/test_data_label_phd.csv --val_label_path=../data/val_data_label_phd.csv --all_file_path=../data/phd_dataset/all_file/




###***************** phd dataset balanced ********************
###-------------> input_size=102400 <-------------
### train Avastnet
#python train_models.py --model_name=AvastNet --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset_balanced/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset_balanced/inputsize_102400/ --train_label_path=../data/train_data_label_phd_balanced.csv --test_label_path=../data/test_data_label_phd_balanced.csv --val_label_path=../data/val_data_label_phd_balanced.csv --all_file_path=../data/phd_dataset/all_file/
#
### train malconv
#python train_models.py --model_name=malconv --input_size=102400 --window_size=500 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset_balanced/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset_balanced/inputsize_102400/ --train_label_path=../data/train_data_label_phd_balanced.csv --test_label_path=../data/test_data_label_phd_balanced.csv --val_label_path=../data/val_data_label_phd_balanced.csv --all_file_path=../data/phd_dataset/all_file/
#
### train fireeye
#python train_models.py --model_name=fireeye --input_size=102400 --window_size=512 --batch_size=32 --epochs=50 --lr=0.0001 --num_workers=1 --log_file_path=../result/phd_dataset_balanced/inputsize_102400/train_log.txt --checkpoint_dir=../checkpoint/phd_dataset_balanced/inputsize_102400/ --train_label_path=../data/train_data_label_phd_balanced.csv --test_label_path=../data/test_data_label_phd_balanced.csv --val_label_path=../data/val_data_label_phd_balanced.csv --all_file_path=../data/phd_dataset/all_file/

