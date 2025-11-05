# IONA Pose Estimation

A real-time 6D pose estimation system combining NVIDIA's FoundationPose and Meta's Detectron2 for robotic manipulation and object tracking applications.

## Overview

This project integrates two powerful frameworks:
- **FoundationPose**: NVIDIA's state-of-the-art 6D object pose estimation
- **Detectron2**: Meta's object detection and segmentation framework

Together, they enable robust real-time tracking of objects in 3D space using RGB-D camera input.

## Prerequisites

- **Operating System**: Linux (tested on Ubuntu 22.04)
- **Python**: 3.9
- **CUDA**: 11.8.0
- **Conda**: Anaconda or Miniconda
- **Hardware**: NVIDIA GPU with CUDA support, Intel RealSense camera (recommended)

## Installation

### 1. Create Python Environment

```bash
conda create -n PoseEstimation python=3.9
conda activate PoseEstimation
conda config --set channel_priority flexible
```

### 2. Install Eigen Library

```bash
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/path/to/eigen/under/conda"
```

> **Note**: Replace `/path/to/eigen/under/conda` with the actual path where Eigen is installed in your conda environment. You can find this using `conda list eigen`.

### 3. Install FoundationPose

```bash
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
python -m pip install -r requirements.txt
```

### 4. Install Additional Dependencies

```bash
# Install nvdiffrast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Install PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Install CUDA Toolkit
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit

# Fix setuptools version
pip uninstall setuptools
pip install setuptools==69.5.1
```

### 5. Build C++ Extensions

```bash
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

> **Important**: After running this command, a new `mycpp` folder will be created inside the `FoundationPose` directory. Copy this folder to your project root directory.

```bash
cp -r mycpp ../
cd ..
```

### 6. Install Detectron2

```bash
# Install PyYAML with specific version
python -m pip install pyyaml==5.1

# Clone and install Detectron2
git clone https://github.com/facebookresearch/detectron2
pip install --no-build-isolation -e ./detectron2
```

### 7. Install Camera Support

```bash
pip install pyrealsense2
```

## Model Weights Setup

### FoundationPose Weights

1. Download the pre-trained network weights from [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing)
2. Create a `FP_weights` directory in your project root
3. Extract all downloaded weights into this folder

```bash
mkdir FP_weights
# Place downloaded weights here
```

### Detectron2 Weights

1. Download the Detectron2 model weights (link to be provided)
2. Create a `detectron2_models` directory in your project root
3. Place the weights and update the `model_info.txt` file accordingly

```bash
mkdir detectron2_models
# Place your .pth model files here
# Update model_info.txt with model details
```

## Project Structure

After completing the setup, your project directory should look like this:

```
.
├── datareader.py                    # Data reading utilities
├── estimater.py                     # Pose estimation core logic
├── FP_Utils.py                      # FoundationPose utility functions
├── Utils.py                         # General utility functions
├── run_real_time_short.py          # Real-time execution script
├── run_main.py                      # Main entry point
│
├── detectron2/                      # Detectron2 framework
│   ├── configs/
│   ├── detectron2/
│   ├── tools/
│   └── ...
│
├── detectron2_models/               # Detectron2 pre-trained models
│   ├── medical_objects_fruits.pth
│   └── model_info.txt
│
├── FoundationPose/                  # FoundationPose framework
│   ├── assets/
│   ├── bundlesdf/
│   ├── learning/
│   ├── mycpp/
│   ├── weights/
│   └── ...
│
├── FP_weights/                      # FoundationPose model weights
│   ├── 2023-10-28-18-33-37/
│   └── 2024-01-11-20-02-45/
│
├── mycpp/                           # C++ extensions (copied from FoundationPose)
│   ├── build/
│   ├── include/
│   ├── src/
│   └── CMakeLists.txt
│
├── Meshes/                          # 3D object meshes for tracking
│   └── Sb_Cup/
│
├── Meal_Tray_Scenario/             # Example scenarios and outputs
│   └── output_20251105_163145/
│
└── learning/                        # Training and dataset utilities
    ├── datasets/
    ├── models/
    └── training/
```

## Usage

### Running Real-Time Pose Estimation

1. **Connect your RealSense camera** to your computer

2. **Run the main script**:
   ```bash
   python3 run_main.py
   ```
3. A window will popup running detection on objects hughlightng them.
4. Press **'K'** to capture the scene
5. Wait for processing, after which the pose frame will be updated and press any key to exit.
6. The results would be stored in the outputs folder at ROOT.

## Configuration

Key configuration files to modify based on your setup:

- `detectron2_models/model_info.txt`: Model metadata and class labels
- Camera parameters in `run_main.py`: Adjust resolution, frame rate, etc.
- Detection thresholds in `estimater.py`: Tune for your specific objects

## Tips for Best Results

1. **Lighting**: Ensure consistent, diffuse lighting without harsh shadows
2. **Camera Position**: Mount camera at appropriate height and angle for your scene
3. **Object Meshes**: Provide accurate 3D meshes in the `Meshes/` directory for better tracking
4. **Calibration**: Perform camera calibration for improved accuracy

---

## Results

*(To be added)*

## Acknowledgements

*(To be added)*

## License

This project builds upon:
- [FoundationPose](https://github.com/NVlabs/FoundationPose) - NVIDIA
- [Detectron2](https://github.com/facebookresearch/detectron2) - Meta AI Research

Please refer to their respective licenses for usage terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and support, please open an issue in the repository.
