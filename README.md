# ShuffleNetv2 in PyTorch

An implementation of `ShuffleNetv2` in PyTorch. `ShuffleNetv2` is an efficient convolutional neural network architecture for mobile devices. For more information check the paper:
[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## Usage

Clone the repo:
```bash
git clone https://github.com/Randl/ShuffleNetV2-pytorch
pip install -r requirements.txt
```

Use the model defined in `model.py` to run ImageNet example:
```bash
python imagenet.py --dataroot "/path/to/imagenet/"
```

To run continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results

TODO