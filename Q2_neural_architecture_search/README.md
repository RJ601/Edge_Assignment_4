# Q2 - Hardware-aware Evolutionary Neural Architecture Search (CIFAR-10)

## How to run

### On Google Colab (recommended)
1. Open `Q2_colab_runner.ipynb` in Colab and set the runtime to **T4 GPU**.
2. Run the cells from top to bottom. When asked, upload all `.py` files of this folder.
3. The last cell downloads a zip of the `output` folder.

### On your laptop (CPU only, slow)
1. `pip install -r requirements.txt`
2. Download CIFAR-10 (python version) and extract it so you have `Q2_neural_architecture_search/data/cifar-10-batches-py/`.
   (Or add `--download` to `evolution.py` / `train_best.py`.)
3. Run in this order:
```
python evolution.py
python train_best.py
python export_onnx.py
python onnx_inference.py
python benchmark.py
python plot.py
```
For a quick test on a laptop: `python evolution.py --population 3 --generations 1 --search_epochs 1`
and `python train_best.py --epochs 1`.
Classify a single image: `python onnx_inference.py --image my.png`

## Folder structure
```
Q2_neural_architecture_search/
  config.py          all paths, search-space options, constraints, hyperparameters
  search_space.py    architecture genome, random sampling, mutation (no PyTorch)
  model_builder.py   architecture generator: genome -> PyTorch network
  data.py            CIFAR-10 in memory + GPU augmentation
  hardware.py        measures CPU latency, model size, peak memory; checks constraints
  trainer.py         training / validation loop used by the search and the final training
  evolution.py       the evolutionary search (writes search history + best architecture)
  train_best.py      trains the best architecture from scratch
  export_onnx.py     exports the trained model to ONNX
  onnx_inference.py  ONNX Runtime CPU inference
  benchmark.py       PyTorch vs ONNX Runtime comparison
  plot.py            plots (one per recorded metric)
  Q2_colab_runner.ipynb  Colab notebook that runs everything
  output/            created by the runs:
    search_history.csv / .json   every candidate with its measurements
    best_architecture.json       winner of the search
    best_model.pth, best_model.onnx
    final_training_log.csv
    benchmark.csv
    plots/             val_accuracy, latency_ms, model_size_mb, peak_memory_mb, train_loss, test_accuracy
```

## Hyperparameters and design decisions
**Search space**
| Item | Options |
|---|---|
| Input resolution | 16, 24, 32 (the network resizes the 32x32 image itself) |
| Stem filters | 8, 16, 24, 32 |
| Depth | 3 to 8 blocks |
| Filters per block | 16, 24, 32, 48, 64, 96, 128 |
| Kernel size | 3, 5, 7 |
| Block type | standard conv, depthwise-separable conv |
| Activation | ReLU, ReLU6, Hardswish, SiLU |
| Stride | 1 or 2 (a stride-2 block is changed to 1 if the feature map would get smaller than 2x2) |

**Evolutionary search**
| Item | Value |
|---|---|
| Population | 10 |
| Generations | 6 (after the random initial generation) |
| Parent selection | tournament of 3 |
| Variation | mutation only (1 or 2 random mutations: change resolution, stem, one block field, add block, remove block) |
| Survivors | best 10 of parents + children |
| Duplicates | an architecture is never evaluated twice |
| Fitness | val accuracy (0-1) - 0.05 * latency/limit - 0.05 * size/limit |

**Hardware-aware constraints** (violators are rejected before any training)
| Item | Value |
|---|---|
| Max CPU latency | 10 ms per image |
| Max model size | 2 MB (float32 weights file) |
| Latency measurement | batch of 1, 1 CPU thread, 10 warm-up + 50 timed runs, median |
| Size measurement | size of the saved `state_dict` file |
| Peak memory | recorded only (not a constraint): highest process memory increase during 20 inference runs, sampled every 1 ms with `psutil` |

The 10 ms limit applies to the CPU the search runs on (Colab CPU on Colab), so change `MAX_LATENCY_MS` in `config.py` if you run it elsewhere.

**Training**
| Item | Value |
|---|---|
| Data split | 45,000 train / 5,000 validation (candidates); test set only for the final model |
| Optimiser | AdamW, learning rate 3e-3, weight decay 1e-4, cosine schedule |
| Batch size | 128 |
| Augmentation | random flip, random crop after 4-pixel padding (one offset per batch) |
| Epochs per candidate | 3 |
| Final training | best architecture from scratch, 30 epochs on all 50,000 training images, test accuracy logged every epoch |
| Final checkpoint | weights after the last epoch |
| ONNX | opset 17, input 1x3x32x32 with dynamic batch size, resize included in the graph |
| Metrics recorded | search: val_accuracy, latency_ms, model_size_mb, peak_memory_mb; final: train_loss, test_accuracy |
