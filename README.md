# HEAT

This repository contains the implementation of the paper "Training Cross-Morphology Embodied AI Agents: From Practical Challenges to Theoretical Foundations". The code is based on [modular-rl](https://github.com/huangwl18/modular-rl), with substantial modifications to align with the methods and experiments described in the paper.

## Environment Setup
1. Install CUDA 12.2
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run
```

2. Install cuDNN
```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

## Setting Up Mujoco
1. Download [Mujoco200 linux](https://www.roboti.us/download/mujoco200_linux.zip)
```bash
cd $HOME
wget https://www.roboti.us/download/mujoco200_linux.zip
```

2. Download [Activation key](https://www.roboti.us/file/mjkey.txt)
```bash
cd $HOME
wget https://www.roboti.us/file/mjkey.txt
```

3. Unzip mujoco200_linux.zip
```bash
mkdir $HOME/.mujoco
unzip mujoco200* -d $HOME/.mujoco
mv $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
```

4. Move license to the bin subdirectory of MuJoCo installation
```bash
mv $HOME/mjkey.txt $HOME/.mujoco/.
```

5. Add system lib path
```bash
sudo gedit $HOME/.bashrc
```

and append the following config
```bash
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export CFLAGS="-I$HOME/.mujoco/mujoco200/include"
export LDFLAGS="-L$HOME/.mujoco/mujoco200/lib"
```

including CUDA environment path, the PATH and LD_LIBRARY_PATH should be
```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt
export CFLAGS="-I$HOME/.mujoco/mujoco200/include"
export LDFLAGS="-L$HOME/.mujoco/mujoco200/lib"
```

6. Test Installation
```bash
source $HOME/.bashrc
cd $HOME/.mujoco/mujoco200/bin
./simulate ../model/humanoid.xml
```

## Clone Project
```bash
cd $HOME
git clone https://github.com/airs-admin/heat.git
```

## Install Dependencies
1. Setting up the conda environment
* Install [Anaconda](https://docs.anaconda.com/anaconda/install/)

2. Create environment
```bash
source $HOME/miniconda3/bin/activate
conda create -c conda-forge python=3.9 -n modular_rl
```

3. Activate modular_rl
```bash
conda activate modular_rl
```

4. Install Torch 2.5.1 with CUDA 12.1
```bash
(modular_rl) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Install as requirements
```bash
(modular_rl) pip install -r requirements.txt
```

6. Install baselines 0.1.5
```bash
(modular_rl) pip install baselines
```

7. If you encounter the following error such as: “ModuleNotFoundError: No module named ‘lockfile’”, the lockfile module however is included in the requirements.txt list. Please install it using the following command
```bash
(modular_rl) pip install lockfile==0.12.2
```

8. Other issues 
* mujoco py install error - fatal error: GL/osmesa.h: No such file or directory
```bash
(modular_rl) sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

* No such file or directory: 'patchelf' on mujoco-py installation
```bash
(modular_rl) sudo apt-get install patchelf
```

9. To run evaluation 
```bash
(modular_rl) pip install matplotlib
```

## Increase Swap Memory Size
```bash
sudo swapoff /swapfile
sudo fallocate -l 250G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Enable permanently
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Run
1. Create morphologies
```bash
(modular_rl) bash run_create_xmls.sh 
```

2. The folder organization should be as follows
```plaintext
heat/
└── src/
    └── environments/
        ├── xmls_cheetah_10/
        │   ├── file1.xml
        │   ├── file2.xml
        │   └── ...
        ├── xmls_cheetah_100/
        │   ├── file1.xml
        │   ├── file2.xml
        │   └── ...
        └── xmls_cheetah_1000/
            ├── file1.xml
            ├── file2.xml
            └── ...
```

3. Run training script
```bash
bash run_eval_type_morphologies.sh
```

4. The custom configurations are as follows
  - MORPHLOGIES: specifies the type of morphology (robot or agent) used in the experiment, e.g. cheetah, hopper
  - CUSTOM_XMLS: a list of custom XML environment configurations for different experiment setups, e.g. ("xmls_hopper_10"), ("xmls_hopper_10, xmls_hopper_100")
  - MAX_NUM_EXP: defines how many models (experiments) should be trained for each XML setting
  - MAX_TIMESTAMPS: specifies the maximum number of timesteps for training each model
  - PARALLEL_COUNT: specifies the number of experiments that should run simultaneously