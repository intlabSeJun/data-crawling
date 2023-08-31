# Diversity Enhancement Network (DEN) 
### 1. Prepare the datasets.
- (1) SYSU-MM01 Dataset [1]: The SYSU-MM01 dataset can be downloaded from this [website](https://github.com/wuancong/SYSU-MM01).
   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.
     
- (2) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.
   - Search for "DBPerson-Recog-DB1" on their website. It is number of 76. You can request the Generated images.

### 2. Training.
  Train a model by
  ```bash
# SYSU-MM01 dataset
python train.py --dataset sysu --data_path ~/SYSU_MM01 --gpu 0 --model all
python train.py --dataset sysu --data_path ~/SYSU_MM01 --gpu 0 --model indoor

# RegDB dataset
python train.py --dataset regdb --data_path ~/RegDB_original --gpu 0 --V_I_flag V_I
python train.py --dataset regdb --data_path ~/RegDB_original --gpu 0 --V_I_flag I_V
```

You may need mannully define the data path.

**Parameters**: More parameters can be found in the script.


### 3. Testing.
Test a model on SYSU-MM01 or RegDB dataset by 
   ```bash
# SYSU-MM01 dataset
python test.py --dataset sysu --data_path ~/SYSU_MM01 --gpu 0 --model all --model_path checkpoint_path
python test.py --dataset sysu --data_path ~/SYSU_MM01 --gpu 0 --model indoor --model_path checkpoint_path

# RegDB dataset
python test.py --dataset regdb --data_path ~/RegDB_original --gpu 0 --V_I_flag V_I --model_path checkpoint_path
python test.py --dataset regdb --data_path ~/RegDB_original --gpu 0 --V_I_flag I_V --model_path checkpoint_path
```

###  5. References.
[1] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.
