# Audio2Image
This is the course project for CSC2541 Generative AI in Machine Learning at University of Toronto

## Venv
Currently using venv in /w/246

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

## Method to use Slurm Cluster
After ssh into comps0.cs.toronto.edu from cs.toronto.edu, run:

```
source /w/247/jameschen/Audio2Image/venv/bin/activate

srun --partition=gpunodes -c 1 --mem=16G --gres=gpu:rtx_4090:1 -t 5-0 --pty <bash_name.sh>
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

'''
drives used: 247-james, 284-james, 284-jerry, 331-eric
