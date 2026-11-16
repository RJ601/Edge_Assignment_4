# Q1 - Knowledge Distillation (ResNet-50 teacher -> MobileNetV2 / SqueezeNet / EfficientNet-Lite0, CIFAR-100)

## How to run

### On Google Colab (recommended)
1. Open `Q1_colab_runner.ipynb` in Colab and set the runtime to **T4 GPU**.
2. Run the cells from top to bottom. When asked, upload all `.py` files of this folder.
3. The last cell downloads a zip of the `output` folder.

### On your laptop (CPU only, very slow for training)
1. `pip install -r requirements.txt`
2. Download CIFAR-100 (python version) and extract it so you have `Q1_knowledge_distillation/data/cifar-100-python/`.
   (Or add `--download` to any command below.)
3. Run in this order:
```
python train_teacher.py
python train_baseline.py
python train_distill.py
python export_onnx.py
python inference.py --model all --variant both
python onnx_inference.py --model all --variant both
python benchmark.py
python plot.py
```
For a quick test on a laptop, add `--epochs 1` to the three training scripts.
Classify a single image: `python inference.py --model mobilenetv2 --variant kd --image my.png`
(`onnx_inference.py` takes the same options).

## Folder structure
```
Q1_knowledge_distillation/
  config.py            all paths and hyperparameters
  data.py              CIFAR-100 loaders and transforms
  models.py            ResNet-50 teacher + the 3 student networks
  distillation.py      KD loss (soft + hard + feature)
  train_utils.py       generic training loop shared by the 3 training scripts
  train_teacher.py     trains the teacher
  train_baseline.py    trains students WITHOUT distillation
  train_distill.py     trains students WITH distillation
  inference.py         unified PyTorch inference for all students
  export_onnx.py       exports students to ONNX
  onnx_inference.py    ONNX Runtime CPU inference
  benchmark.py         accuracy / size / latency table
  plot.py              plots (train loss, test accuracy)
  Q1_colab_runner.ipynb  Colab notebook that runs everything
  output/              created by the runs:
    checkpoints/       teacher_resnet50.pth, baseline_*.pth, kd_*.pth
    logs/              per-epoch CSV for every run
    onnx/              baseline_*.onnx, kd_*.onnx
    plots/             train_loss.png, test_accuracy.png
    benchmark.csv
```

## Hyperparameters and design decisions
| Item | Value |
|---|---|
| Input size | CIFAR images up-sampled from 32x32 to 64x64 |
| Initialisation | ImageNet-pretrained weights for all networks (`--no_pretrained` to disable) |
| EfficientNet-Lite0 | from the `timm` library (not in torchvision) |
| Augmentation | RandomCrop(32, pad 4), horizontal flip |
| Optimiser | SGD, momentum 0.9 (Nesterov), weight decay 5e-4, cosine LR schedule |
| Learning rate | 0.02 (ResNet-50, MobileNetV2, EfficientNet-Lite0), 0.01 (SqueezeNet) |
| Batch size | 128 |
| Epochs | 15 for teacher, 15 for every baseline and every distilled student |
| Mixed precision | on when a GPU is available |
| Temperature T | 4.0 |
| Loss weights | total = 0.7 * soft + 0.3 * hard + 1.0 * feature |
| Soft loss | KL divergence between softened distributions, multiplied by T^2 |
| Feature loss | MSE between the teacher's last feature map and the student's last feature map after average-pooling to the teacher's spatial size and a trainable 1x1 conv (training only, not exported) |
| SqueezeNet head | final ReLU removed so logits can be negative like the teacher's |
| Checkpoint | weights after the last epoch (no picking the best epoch on the test set) |
| Baseline vs KD | same seed, same recipe; the only difference is the loss |
| ONNX | opset 17, dynamic batch size, CPU inference with ONNX Runtime |
| Latency benchmark | batch of 1, 1 CPU thread for PyTorch and ONNX Runtime, 20 warm-up + 100 timed runs, median reported |
| Metrics logged | train_loss (hard-label cross-entropy, so KD and baseline curves are comparable), test_accuracy |
