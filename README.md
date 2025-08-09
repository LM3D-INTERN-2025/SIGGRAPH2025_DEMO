# Lumio3D HEAD AVATAR DEMO 2025

- This demo makes changes to [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars) in order to be fully compatible with Lumio Scanner.
- More

## To use this demo

Clone the project

```bash
  git clone https://github.com/LM3D-INTERN-2025/SIGGRAPH2025_DEMO.git --recursive 
  cd SIGGRAPH2025_DEMO
```

## Installation

Our default installation method is based on Conda package and environment management:

### 1. Create conda environment and install CUDA

```shell
conda create --name ga -y python=3.10
conda activate ga

# Install CUDA and ninja for compilation
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja  # use the right CUDA version
```

### 2. Setup paths

#### For Linux

```shell
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"
conda env config vars set CUDA_HOME=$CONDA_PREFIX  # for compilation

# you might have to re-activate the environment here.
conda deactivate
conda activate ga
```

#### For Windows with PowerShell

```shell
conda env config vars set CUDA_PATH="$env:CONDA_PREFIX"  

## Visual Studio 2022 (modify the version number `14.39.33519` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;$env:PATH"
## or Visual Studio 2019 (modify the version number `14.29.30133` accordingly)
conda env config vars set PATH="$env:CONDA_PREFIX\Script;C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;$env:PATH" 

# re-activate the environment to make the above eonvironment variables effective
conda deactivate
conda activate ga
```

### 3. Install PyTorch and other packages

```shell
# Install PyTorch (make sure that the CUDA version matches with "Step 1")
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# make sure torch.cuda.is_available() returns True

# Install the rest packages (can take a while to compile diff-gaussian-rasterization, simple-knn, and nvdiffrast)
cd GaussianAvatars
pip install -r requirements.txt
cd ..
```

### 4. Install pytorch3d
```shell
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# or other installation process via https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
```

#### and you should be good to go.