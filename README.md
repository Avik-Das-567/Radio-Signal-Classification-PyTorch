# Radio Signal Classification with PyTorch and EfficientNet

## Overview

This project builds a PyTorch image-classification pipeline for identifying radio signal types from spectrogram images. Each input example is represented as a flattened spectrogram, reshaped into a single-channel image, augmented with spectrogram-specific transformations, and classified with a pretrained EfficientNet-B0 convolutional neural network.

The classification task covers four radio signal categories:

| Label | Encoded class |
| --- | ---: |
| `Squiggle` | 0 |
| `Narrowband` | 1 |
| `Narrowbanddrd` | 2 |
| `Noises` | 3 |

The notebook trains the model on spectrogram data stored in CSV format, evaluates it against a held-out validation set, saves the best model weights by validation loss, and performs inference on random validation examples with class-probability visualizations.

## Project Workflow

1. Import the scientific Python, PyTorch, torchvision, and `timm` dependencies used for data manipulation, model definition, training, and visualization.
2. Configure the experiment with CSV paths, batch size, compute device, model name, learning rate, and epoch count.
3. Load `train.csv` and `valid.csv` into pandas DataFrames.
4. Inspect the number of examples, identify the available labels, and visualize sample spectrograms by reshaping flattened pixel vectors into `64 x 128` images.
5. Define SpecAugment-style transformations for masking regions along the time and frequency axes of each spectrogram.
6. Build a custom `SpecDataset` that maps string labels to integer targets, reshapes flattened spectrogram vectors, converts them into PyTorch tensors, and applies augmentations to training samples.
7. Wrap the training and validation datasets in PyTorch `DataLoader` objects for batched iteration.
8. Load a pretrained EfficientNet-B0 model from `timm`, adapt it for single-channel spectrogram inputs, and replace the classifier head for four output classes.
9. Define training and evaluation functions that compute cross-entropy loss and multiclass accuracy for every epoch.
10. Train for 15 epochs with Adam optimization, validating after each epoch and saving the model checkpoint whenever validation loss improves.
11. Reload the best saved weights and run inference on random validation-set spectrograms, displaying the input image beside the predicted class-probability distribution.

## Dataset Representation

The dataset is loaded from two CSV files:

| Split | Notebook path | Examples | Batches at `batch_size=128` |
| --- | --- | ---: | ---: |
| Training | `/content/train.csv` | 3,200 | 25 |
| Validation | `/content/valid.csv` | 800 | 7 |

Each row contains `8,192` numeric pixel values plus a `labels` column. The pixel values are read from columns `0:8192`, converted to `float64` with NumPy, and resized into a spectrogram image of shape `64 x 128`.

Inside the dataset class, each spectrogram is reshaped to `64 x 128 x 1`, converted into a PyTorch tensor, and permuted to channel-first format:

```text
flattened CSV row -> (64, 128, 1) -> (1, 64, 128)
```

After batching, the model receives tensors with shape:

```text
images: torch.Size([128, 1, 64, 128])
labels: torch.Size([128])
```

This layout treats each spectrogram as a grayscale image where the two spatial dimensions represent the time-frequency structure of the radio signal.

## Spectrogram Augmentation

The project uses SpecAugment-style masking to improve robustness during training. The notebook imports `TimeMask` and `FreqMask` from `spec_augment.py` and composes them with `torchvision.transforms.Compose`.

The active training augmentation pipeline is:

```python
T.Compose([
    TimeMask(T=15, num_masks=4),
    FreqMask(F=15, num_masks=3)
])
```

`TimeMask` randomly masks vertical time spans across the spectrogram. For each mask, it samples a width up to `T`, selects a start position along the time axis, and replaces the selected region with either zeros or the spectrogram mean. In this notebook, mean replacement is used because `replace_with_zero` defaults to `False`.

`FreqMask` applies the same idea along the frequency axis. It samples a frequency-band width up to `F`, chooses a band location, and fills that region with either zeros or the spectrogram mean.

The helper module also defines `TimeWarp`, which performs nonlinear time-axis deformation through sparse image warping. That class is implemented but is not included in the notebook's active `get_train_transform()` pipeline. Its warping operation depends on `sparse_image_warp.py`, which implements dense flow generation, polyharmonic spline interpolation, and bilinear image sampling.

## Custom PyTorch Dataset

The notebook defines `SpecDataset`, a subclass of `torch.utils.data.Dataset`, to bridge the CSV representation and the CNN input format.

The dataset is responsible for:

- Storing a pandas DataFrame for a given split.
- Mapping string labels into integer class IDs.
- Extracting the first `8,192` columns as spectrogram pixels.
- Resizing each flattened row into a `64 x 128 x 1` image.
- Converting the image into a channel-first tensor.
- Applying augmentations only when an augmentation pipeline is provided.
- Returning `(image.float(), label)` for each index.

Training data is instantiated with the augmentation pipeline, while validation data is instantiated without augmentation. This keeps validation metrics tied to the original spectrogram distribution rather than randomly masked variants.

## Model Architecture

The model is defined as a lightweight wrapper around a pretrained EfficientNet-B0 backbone from the `timm` model library:

```python
timm.create_model(
    "efficientnet_b0",
    num_classes=4,
    pretrained=True,
    in_chans=1
)
```

The important architectural adaptation is `in_chans=1`, which allows EfficientNet-B0 to consume grayscale spectrogram tensors instead of standard three-channel RGB images. The classifier output dimension is set to `4`, matching the four radio signal classes.

The forward pass returns logits during inference. During training or evaluation, when labels are supplied, it also computes `nn.CrossEntropyLoss()` directly inside the model wrapper:

```text
images -> EfficientNet-B0 -> class logits -> cross-entropy loss
```

This design keeps the training and validation loops concise because each batch can retrieve both logits and loss from the model call.

## Training and Evaluation Strategy

The notebook uses the following training configuration:

| Parameter | Value |
| --- | --- |
| Model | `efficientnet_b0` |
| Pretrained weights | Enabled |
| Input channels | 1 |
| Output classes | 4 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Epochs | 15 |
| Device | `cpu` |
| Loss | Cross-entropy |
| Metric | Multiclass accuracy |

The training function sets the model to training mode, iterates over `trainloader`, moves images and labels to the configured device, clears gradients, computes logits and loss, backpropagates, and updates the model parameters. Running loss and accuracy are accumulated across batches and displayed with `tqdm`.

The evaluation function sets the model to evaluation mode and wraps validation in `torch.no_grad()` to avoid gradient tracking. It uses the same loss and accuracy calculations as the training function, but does not update model weights.

Accuracy is computed by `multiclass_accuracy()` in `utils.py`. The function selects the top predicted class with `topk(1)`, compares it with the ground-truth label tensor, casts matches to floating-point values, and returns the mean batch accuracy.

The training loop tracks `best_valid_loss`, initialized to infinity. After every epoch, the model is saved only if the validation loss improves:

```text
efficientnet_b0-best-weights.pt
```

## Training Results

The model was trained for 15 epochs. The best checkpoint was selected by validation loss, not by the final epoch.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1.243365 | 0.704063 | 3.171487 | 0.299107 | Saved |
| 2 | 0.426498 | 0.757187 | 2.599505 | 0.433036 | Saved |
| 3 | 0.386398 | 0.764375 | 0.594445 | 0.661830 | Saved |
| 4 | 0.415021 | 0.744375 | 0.374673 | 0.772321 | Saved |
| 5 | 0.370643 | 0.762500 | 0.409013 | 0.699777 | - |
| 6 | 0.355402 | 0.768750 | 0.376337 | 0.750000 | - |
| 7 | 0.351281 | 0.785000 | 0.391677 | 0.766741 | - |
| 8 | 0.349440 | 0.793437 | 0.408925 | 0.756696 | - |
| 9 | 0.357876 | 0.790000 | 0.448666 | 0.748884 | - |
| 10 | 0.369994 | 0.783437 | 0.435265 | 0.753348 | - |
| 11 | 0.321411 | 0.817500 | 0.473461 | 0.719866 | - |
| 12 | 0.331040 | 0.822812 | 0.492522 | 0.725446 | - |
| 13 | 0.292621 | 0.842500 | 0.543607 | 0.756696 | - |
| 14 | 0.294203 | 0.851875 | 0.605699 | 0.698661 | - |
| 15 | 0.269116 | 0.857813 | 0.688842 | 0.694196 | - |

The strongest validation result occurred at epoch 4:

| Metric | Value |
| --- | ---: |
| Best validation loss | 0.374673 |
| Best validation accuracy | 0.772321 |
| Final training accuracy | 0.857813 |
| Final validation accuracy | 0.694196 |

Training accuracy continued to rise through the final epoch, while validation loss increased after the best checkpoint. This makes the saved epoch-4 weights the most appropriate model state for inference according to the notebook's validation-loss criterion.

## Inference Pipeline

The notebook performs inference after training by reloading the best checkpoint:

```python
model.load_state_dict(torch.load(MODEL_NAME + "-best-weights.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()
```

For each inference example, a random index is sampled from the validation dataset. The spectrogram tensor is expanded with a batch dimension, passed through the model, and converted from logits to probabilities with softmax:

```text
(1, 64, 128) -> unsqueeze -> (1, 1, 64, 128) -> logits -> softmax probabilities
```

The `view_classify()` helper in `utils.py` displays two panels:

- The input spectrogram image.
- A horizontal bar chart of class probabilities.

This provides a qualitative check of the trained classifier by pairing each validation spectrogram with its predicted probability distribution across the four radio signal categories.

## Technical Notes

The project combines transfer learning with spectrogram-specific augmentation. EfficientNet-B0 supplies a compact pretrained CNN backbone, while the masking transforms expose the model to partially occluded time-frequency regions during training.

The helper files support the notebook as follows:

- `utils.py` implements multiclass accuracy and prediction visualization.
- `spec_augment.py` implements time masking, frequency masking, and an optional time-warp transform.
- `sparse_image_warp.py` implements the interpolation and dense warping utilities required by the optional time-warp transform.

The active notebook path uses time and frequency masking only. Time warping is available in the helper code but is not applied in the configured training transform.

The validation split is evaluated without augmentation, so validation metrics reflect model performance on unmodified spectrogram examples. The saved checkpoint is based on minimum validation loss, which is why the best model is selected from epoch 4 even though training accuracy is highest at epoch 15.
