# Super-Resolution Image Reconstruction based on Random-coupled Neural Network and EDSR

## Introduction

This project implements the EDSR-RCNN model, a deep learning approach that exhibits strong performance in super-resolution reconstruction. The model employs the Random-coupled Neural Network (RCNN) [1] to efficiently extract mid- to high-scale feature information from images, eliminating the need for complex training procedures or costly deep neural architectures. By integrating these spatiotemporal features, the EDSR-RCNN model enhances the convolutional kernel scale of the Enhanced Deep Residual Network (EDSR) [2], thereby facilitating the extraction of more intricate image features.

If you find our work useful in your research or publication, please cite our work:

Haoran Liu, Kaimin Wang, Mingzhe Liu, Xue Zuo, Peng Li, and Ruili Wang "Super-Resolution Image Reconstruction based on Random-coupled Neural Network and EDSR." Not published yet. 

![](README.png)


This project is based on the EDSR project "https://github.com/sanghyun-son/EDSR-PyTorch" The specific structure and functional implementation of the EDSR model developed can be found in the article:

[1] Bee Lim,  Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee,  "Enhanced Deep Residual Networks for Single Image Super-Resolution," 2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with CVPR 2017. 

## Dependencies
* Python 3.6

* PyTorch >= 1.0.0

* numpy

* skimage

* imageio

* matplotlib

* tqdm

* PIL

* cv2 >= 3.xx (Only if you want to use video input/output)

## Code:
   You can directly run the **`demo.sh`** in the src folder or use terminal commands to conduct experiments.
   
   For example:

   ```bash
   cd src       
   sh demo.sh
   ```

## How to train and test EDSR-RCNN： 

* As we employ the DIV2K dataset for training and testing our model, please download the dataset from the links available in the dataset and model results section at the end of this document. Several important considerations must be taken into account during the training and testing processes:

* Set the **`--dir_data`** parameter in **`option.py`** to the path where the DIV2K dataset is located. If there are specific requirements for storing the dataset, modify the **`--dir_data`** setting to reflect the specified path.

* To train the model on alternate datasets, adjust **`--data_train`** and **`--data_test`** to correspond with the names of those datasets.

* If you intend to continue training from a pre-trained model, set **`--pre_train`** to the path of that model.

* The RCNN channel of the model is enabled by default. To train the original EDSR model, set **`--RCNN_channe`** to off.

* To test a trained model exclusively, add **`--test_only`**.

* For adjusting relevant parameters of the EDSR model, specify them before running **`demo.sh`**.

* We have pre-optimized a set of RCNN model parameters suitable for the DIV2K dataset; therefore, these parameters do not require modification. If you intend to train the EDSR-RCNN model on different datasets, please adjust the RCNN model parameters accordingly beforehand.
## Usage
1. You should first construct the dataset folder according to a specific structure.
   ```
   dataset/
   └── DIV2K/
       ├── DIV2K_train_HR/
       │   ├── 0001.png
       │   ├── 0002.png
       │   └── 0003.png
       ├── DIV2K_train_LR_bicubic/
       │   └── X2/
       │       ├── 0001x2.png
       │       ├── 0002x2.png
       │       └── 0003x2.png
   ```

2. You can enter specified commands in the **`demo.sh`** file according to your own needs to build the training and testing process.
   For example, if you want to train a model on a certain dataset, the training set is the first seven hundred images, the testing set is the last hundred images, the number of residual blocks in the model is 32, and the residual linking factor is 0. 1. The number of feature maps is 100, and the loss type is selected as smoothl1loss. You can run the following line of code.
   ```bash
   python main.py --model EDSR --scale 2 --data_range 1-700/701-800 --save your_path --n_colors 1 --n_resblocks 32 --res_scale 0.1  --loss 1*SmoothL1Loss --reset --n_feats 100
   ```
   
3. If you only want to test on the trained model, you need to add **`-- test_only`** and specify where the image to be tested is in the dataset. For example, to test the last two images of the DIV2K dataset on the trained model, you can run the following code:
   ```bash
   python main.py --model EDSR --scale 2 --data_range 1-700/799-800 --save your_path --n_colors 1 --n_resblocks 32 --res_scale 0.1  --loss 1*SmoothL1Loss --reset --n_feats 100 --test_only --pre_train 'your_model_path'
   ``` 
5. You can find the result images from **`experiment/test/results/your_path`**.
## Optuna parameter tuning
Optuna is an open-source automated hyperparameter optimization framework designed to help developers and researchers efficiently tune the hyperparameters of machine learning models [2]. It offers a simple yet powerful way to find the best combinations of hyperparameters, thereby enhancing model performance.

You can directly run the **`optuna_utility.py`** file to adjust parameters. If there are other parameters that need to be adjusted, they can be specified in the **`objective`** function, taking the learning rate as an example.
   ```bash
   lr_star = trial.suggest_float("lr_star", 1e-6, 1e-4, log=True)
   ```

[2] Akiba, Takuya, et al. "Optuna: A next-generation hyperparameter optimization framework." Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2019.

## Dataset and model results

This application will open source all trained models and test results, as well as the DIV2K dataset, which you can download here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13340845.svg)](https://doi.org/10.5281/zenodo.13340845).
