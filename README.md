# SpinQuant
## Install:
Install Pytorch. 

Install requirement.txt

It is important to use the specified transformers version, because we inherit and overwrite Trainer's init function.

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git

cd fast-hadamard-transform

pip install .

## Flags:

```--checkpoint_local_path``` load R.bin rotation checkpoint during evaluation

For training, use ```local_ptq.sh```. For eval, use ```local_eval.sh```



