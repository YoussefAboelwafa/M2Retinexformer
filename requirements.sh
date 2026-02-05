conda create -n m2retinexformer python=3.9 -y

conda activate m2retinexformer

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm nvitop

python setup.py develop --no_cuda_ext