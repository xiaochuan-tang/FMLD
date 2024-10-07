This is the source code for the paper "FMLD: Vertical Federated Learning for Multi-Modal Landslide Detection".

# FMLD User Guide

## System Requirements

- **Operating System**: Linux
- **Hardware Requirements**: At least 80GB GPU memory is recommended.
- **Software Dependencies**: Python 3.8, CUDA 11.3

## Environment Setup

1. Clone the repository:

   ```bash
   cd FMLD
   ```

2. Create and activate a virtual environment.

3. Install the required dependencies:

   The experiment framework relies on PaddleSeg. For detailed configuration requirements, refer to: [PaddleSeg Installation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9.1/docs/install.md)

## Dataset Preparation

Download the datasets and organize the files as follows:

```kotlin
// Datasets are stored in the dataset directory
// Jiuzhaigou Dataset
dataset/
  ├── dataset_jiuzhaigou/
  │   ├── train_hill_jpeg/     // hillshade data
  │   ├── train_opt_jpeg/      // opt data
  │   ├── train_dem_tif/       // dem data
  │   ├── train_label_png/     // label data
  │   ├── train_new.txt        // training set file list
  │   └── test_new.txt         // test set file list

// Luding Dataset
  ├── dataset_luding/
  │   ├── train_hill_jpeg/     // hillshade data
  │   ├── train_opt_jpeg/      // opt data
  │   ├── train_dem_tif/       // dem data
  │   ├── train_label_png/     // label data
  │   ├── train.txt            // training set file list
  │   └── test.txt             // test set file list
```

A sample dataset is available in the directory /FMLD/dataset.

## Model preparation

```kotlin
// Models are stored in the output directory
// Jiuzhaigou Models
output/
  ├── jiuzhaigou/
  │   ├── server/
  │   │   └── best_model/
  │   │       └── model.pdparams         // The best model saved on the server
  │   ├── client1/
  │   │   └── segformer_opt.pdparams     // Model parameters for optical segmentation
  │   ├── client2/
  │   │   └── hrnet_dem.pdparams         // Model parameters for DEM segmentation
  │   └── client3/
  │       └── hrformer_base_hillshade.pdparams  // Model parameters for hillshade segmentation

// Luding Models
  ├── luding/
  │   ├── server/
  │   │   └── best_model/
  │   │       └── model.pdparams         // The best model saved on the server
  │   ├── client1/
  │   │   └── segformer_opt.pdparams     // Model parameters for optical segmentation
  │   ├── client2/
  │   │   └── hrnet_dem.pdparams         // Model parameters for DEM segmentation
  │   └── client3/
  │       └── hrformer_base_hillshade.pdparams  // Model parameters for hillshade segmentation

  ├── pretrained/
  │   ├── hrnet_dem.pdparams/
```



## Train

To reproduce the main results, run:

```bash
# Jiuzhaigou
bash sh/jiuzhaigou/train.sh

# Luding
bash sh/luding/train.sh
```

## Val

After running the code, results will be saved in the `output/` directory. To compare with the results in the paper, use the evaluation scripts:

```bash
# Jiuzhaigou
bash sh/jiuzhaigou/val.sh 

# Luding
bash sh/luding/val.sh
```

## Predict

```bash
# Jiuzhaigou
bash sh/jiuzhaigou/predict.sh

# Luding
bash sh/luding/predict.sh
```

## Attention

1. This experiment requires launching one server and three clients, totaling four terminal commands to start the model. To facilitate the reproduction of the experimental results from the paper, all commands are integrated into the sh scripts, and only the server's log content will be output.
2. If you need to view the client log contents or encounter errors while running the code, you can open four terminals to run the Python scripts separately for detailed information.

