# Deep Colorization

A basic convolutional neural network (CNN) approach to image colorization.

## Requirements

Keras, TensorFlow, Scikit-Learn.

## What is Colorization?

Colorization refers to the task of taking a grayscale image as input and producing a "realistically" colored version of the input image as output.

## CNNs for Colorization

The implemented model is roughly based off of the model described [here](http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/). Essentially, it consists of a series of convolutional layers followed by alternating upsampling and convolutional layers.

## Running the Model

The following will generate colorized versions of the provided image dataset:

```python
python deep_colorization.py
```



