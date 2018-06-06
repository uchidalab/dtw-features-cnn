# Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks

This is the repository for the replication of "Introducing Local Distance-Based Features to Temporal Convolutional Neural Networks" by B. K. Iwana, M. Mori, A. Kimura, and S. Uchida.

## Requirements
Python >= 3.0

TensorFlow >= 1.4

Numpy >= 1.13.3

scikit-learn 0.19.1 (for test files only)

## Execution

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

## Model Details

### Network Type

### Convolution Type


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
