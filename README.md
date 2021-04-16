==================================================================
Software Abstract

The software I am working with is the pytorch_connectomics package that was created by the Visual Computing Group at Harvard. It is a machine learning package that primarily does image segmentation tasks. Currently, my job involves using it to automatically label electron microscope images of plants. Labelling the organelles in plants is a very time consuming task, so it is near impossible to do it with an extremely large dataset. However, this type of machine learning should allow us to label only a fraction of a dataset, have the model learn how to label, and then automatically label the rest. There are two main things I am interested in speeding up.

1) While training the model is a one time cost I am not worried about improving, I think model prediction can be sped up. This to some extend is a pleasantly parallel task, so I should be able to use a slurm job array to divide up a large image stack into segments and give each node a section of it to label. This should speed up prediction.

2) I have a couple operations that I do on large arrays with numpy that probably could be sped up by using something like numba or cupy. These are mainly preprocessing operations.


==================================================================
To install the necissary requirements, run the following commands (the commands can also be run using ./installScript.sh that is in my github. It will create a projectFolder wherever the working directory is)

```
module load python

module load CUDA

mkdir projectFolder

cd projectFolder

git clone https://github.com/zudi-lin/pytorch_connectomics.git

git clone https://github.com/brookho8/CMSE401Project.git

cd pytorch_connectomics

python3 -m venv projectEnvironment

source projectEnvironment/bin/activate

pip3 install -r requirements.txt

pip3 install --editable .

pip install torch torchvision torchaudio

cp ../CMSE401Project/submission.sb submission.sb

cp ../CMSE401Project/example.py example.py

cp ../CMSE401Project/config.yaml configs/MyConfigProject.yaml
```

=====================================

Also, please use the interactive desktop to download the following files into the pytorch_connectomics folder


https://drive.google.com/file/d/1ysxsG4dcdMzdcIa7vzuWuo2u5VgGVmD8/view?usp=sharing

https://drive.google.com/file/d/1JRmnIcCaFjTiw29vrA572R8ZUk1Tx2M2/view?usp=sharing

https://drive.google.com/file/d/1G-Fw6DEMrUPGS2s2XPSWJpCT2yXu4ow8/view?usp=sharing



==================================================================

I included some example code in example.py of an array operation I may be able to speed up. You can run and time this example using the command "time python3 example.py"


==================================================================

There is a submission script that should ask for the proper resources (including GPU) to train and then do a prediction using the neural network and some example data. A submission script is used for this since it is resource intensive and also takes a long time. If the above commands are all run you should be able to submit the job with the command "sbatch submission.sb"


==================================================================

The pytorch_connectomics github can be found at https://github.com/zudi-lin/pytorch_connectomics
The other example was code I wrote. I will be benchmarking more examples in depth for the final report.
