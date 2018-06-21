# Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks

This is the repository for the replication of "Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks" by B. K. Iwana, M. Mori, A. Kimura, and S. Uchida.

## Table of Contents

1. [Model Details](#model-details)
2. [Requirements](#requirements)
3. [How to Use](#how-to-use)
4. [Citation](#citation)

## Model Details

### DTW Features
![DTW Features](https://github.com/uchidalab/dtw-features-cnn/blob/master/readme/dtwfeatures.PNG "DTW Features")

### Network Type
![Network Types](https://github.com/uchidalab/dtw-features-cnn/blob/master/readme/multimodal.PNG "Network Types")

### Convolution Type
![Convolution Types](https://github.com/uchidalab/dtw-features-cnn/blob/master/readme/convolutiontype.PNG "Convolution Types")

## Requirements
Python >= 3.0

TensorFlow >= 1.4

Numpy >= 1.13.3

scikit-learn 0.19.1 (for test files only)

## How to Use

### Step 1 - Generate Datasets:

Included in the repository are the Unipen 1a, 1b, and 1c datasets. 

To train the models, first you need to generate the DTW features.

```
python3 generate_dataset.py
```

### Step 2 - Train:

To train a network use:

```
python3 cnn-train-[raw|dtwfeatures|earlyfusion|midfusion|latefusion]-[1d|2d].py [dataset] [conv width]
```
Where,
* **\[raw|dtwfeatures|earlyfusion|midfusion|latefusion]** refers to the [Network Type](#network-type)
* **\[1d|2d]** is [1D or 2D Convolutions](#convolution-type)
* **\[dataset]** is either "1a", "1b", or "1c" and refers to the Unipen dataset. 
  * 1a: digits
  * 1b: uppercase characters
  * 1c: lowercase characters.
* **\[conv width]** is the width of the convolution. 
  * For 1D convolutions: **\[conv width]** x dimensionality of the data (Width x Channel). 
  * For 2D convolutions: 1 x **\[conv width]** x dimensionality of the data (Height x Width x Channel).

Example:
```
python3 cnn-train-midfusion-2d.py 1a 3
```
which results in a feature-level fusion (**midfusion**) network with **2D** convolutions using Unipen **1a** (digits) with a convolution size of (**3**, 1)

### Step 3 - Test:

Same parameters as train

Example:
```
python3 cnn-**test**-midfusion-2d.py 1a 3
```

## Citation

B. K. Iwana, M. Mori, A. Kimura, and S. Uchida, "Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks," in *Int. Conf. Frontiers in Handwriting Recognition*, 2018.

```
@article{iwana2018introducing,
  title={Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks
},
  author={Iwana, Brian Kenji and Mori, Minoru and Kimura, Akisato and Uchida, Seiichi},
  booktitle={Int. Conf. Frontiers in Handwriting Recognition},
  year={2018}
}
```
