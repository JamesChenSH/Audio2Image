# Audio2Image
This is the course project for CSC2541 Generative AI in Machine Learning at University of Toronto

## Environment Setup
```
pip install -r requirements.txt
```
Note that pytorchvideo is deprecated and using old pytorch functions, we will 
need to go into pytorchvideo module, change the import file in the script
`<path to library>/pytorchvideo/transforms/augmentation.py`'s 9th line
as following:
```python
import torchvision.transforms.functional as F_t
```


## Dataset
We use the audio - image corresponding datasets here:
https://zenodo.org/records/3828124


## Getting Started
To have a sample of an image from the model we arrived at for final project, run 
```
python stable_diffusion\sample_diffusion_imagebind.py
```
You may add arguments `--[SounDiff_S, SounDiff_F]` to select which model to use.
You can also use `--prompted` argument to add "satellite image" prompt to the model.
The output images are in the `/output_images/` folder.


## Method to use Slurm Cluster
After ssh into comps0.cs.toronto.edu from cs.toronto.edu, run:

```
source /w/247/jameschen/Audio2Image/venv/bin/activate
source /w/284/jerryzhao/Audio2Image/.venv/bin/activate

srun --partition=gpunodes -c 1 --mem=16G --gres=gpu:rtx_4090:1 -t 5-0 --pty <bash_name.sh>
srun --partition=gpunodes -c 1 --mem=16G --gres=gpu:rtx_4090:1 -t 1-0 --pty run_gpu.sh
srun --partition=gpunodes -c 1 --mem=16G --gres=gpu:rtx_a6000:1 -t 1-0 --pty run_gpu.sh

srun --partition=gpunodes -c 1 --mem=32G --gres=gpu:rtx_4090:1 -t 1-0 --pty python stable_diffusion_imagebind.py
srun --partition=gpunodes -c 1 --mem=32G --gres=gpu:rtx_a6000:1 -t 1-0 --pty python stable_diffusion_imagebind.py
```

## About using Mountable Terminal
Creating a new "mountable" terminal
```
tmux new -s <terminal name> (Current name is gpu_job)
```

Then it will create an integrated terminal. To unmount:
press ctrl+b first, then quickly press d, once. 

To re-mount the terminal:
```
tmux attach -t <termial name>
```
