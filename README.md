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

To continue training from checkpoint
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "/path/to/checkpoint/folder"
```
## Results


For x0.5 model I achieved 0.4% lower top-1 accuracy than claimed.

|Classification Checkpoint| MACs (M)   | Parameters (M)| Top-1 Accuracy| Top-5 Accuracy|  Claimed top-1|  Claimed top-5|
|-------------------------|------------|---------------|---------------|---------------|---------------|---------------|
|      [shufflenet_v2_0.5]|41          |1.37           |          59.86|          81.63|           60.3|              -|

You can test it with
```bash
python imagenet.py --dataroot "/path/to/imagenet/" --resume "results/shufflenet_v2_0.5/model_best.pth.tar" -e --scaling 0.5
```
