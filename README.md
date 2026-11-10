# Deep Learning on Fashion-MNIST

Four deep learning projects built in TensorFlow/Keras on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000
grayscale 28×28 images of clothing in 10 classes. They cover classification,
learning from unlabelled data, and two kinds of generative model.

Individual coursework, COMP 263 Deep Learning, Centennial College (Winter 2023).

| # | Notebook | Model | Result |
| --- | --- | --- | --- |
| 1 | [CNN vs LSTM classifier](notebooks/01_cnn_vs_lstm_classifier.ipynb) | CNN and row-by-row LSTM | CNN 88.6% test accuracy, LSTM 86.5% |
| 2 | [Denoising autoencoder + transfer learning](notebooks/02_denoising_autoencoder_transfer_learning.ipynb) | Convolutional denoising autoencoder reused in a classifier | 73.2% → 76.7% accuracy with only 1,800 labels |
| 3 | [Variational autoencoder](notebooks/03_variational_autoencoder.ipynb) | VAE with a 2-D latent space | See known issues |
| 4 | [DCGAN](notebooks/04_dcgan_trousers.ipynb) | Generator and discriminator with a custom training loop | Synthetic trouser images |

## 1. CNN vs LSTM

Compares two ways of reading an image. A CNN (two 32-filter convolution
layers with max-pooling, about 90,700 parameters) reached **88.6%** test
accuracy. An LSTM with 128 units, reading each image as 28 rows over 28 time
steps (about 81,700 parameters), reached **86.5%**. At a similar size, the
CNN's built-in sense of 2-D locality wins.

## 2. Learning from unlabelled data

Simulates a common real problem: plenty of data, few labels. A baseline CNN
trained on **1,800 labelled images** reached **73.2%**. A convolutional
denoising autoencoder was then trained on **57,000 unlabelled images**
(Gaussian noise, factor 0.2) to reconstruct clean images. Reusing it in the
classifier raised test accuracy to **76.7%**, a 3.5-point gain from data
that had no labels at all.

## 3. Variational autoencoder

An encoder outputs a mean and log-variance for a 2-dimensional latent space,
a custom layer samples from it with the reparameterization trick, and a
decoder maps latent points back to images. The latent space is plotted by
class.

## 4. DCGAN

Trained on the 7,000 trouser images, scaled to [-1, 1]. The generator maps a
100-dimensional noise vector to a 28×28 image through transposed convolutions
with batch normalization and LeakyReLU (about 2.3M parameters). Training uses
a custom `GradientTape` loop with separate optimizers for the generator and
discriminator (Adam, learning rate 0.0002, β₁ = 0.5).

## Known issues and next steps

Reviewing this work later, I found problems I plan to fix:

- **Lab 1:** the confusion matrix passes predictions before true labels, so its axes are swapped.
- **Lab 2:** the classifier reuses the whole autoencoder rather than only the encoder. Next step: transfer only the encoder and compare frozen vs fine-tuned layers.
- **Lab 3:** the loss uses only the KL term, with a sign error, and no reconstruction loss, so the model does not learn (the loss stays flat). The decoder also outputs 14×14 instead of 28×28, and the latent plot uses an untrained encoder.
- **Lab 4:** the discriminator ends in a sigmoid while the loss uses `from_logits=True`, applying the sigmoid twice. It also contains an unusual upsampling layer.

## Run

```bash
pip install tensorflow scikit-learn matplotlib seaborn jupyter
jupyter notebook notebooks/
```

The dataset downloads automatically through `tf.keras.datasets.fashion_mnist`.
