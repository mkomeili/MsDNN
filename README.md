1. Build the docker file
docker build -t msdnn .

2. Run the docker file with gpu (make sure you in msdnn folder)
docker run -it --rm --gpus all -v $PWD/code:/home/python_user/app msdnn bash

3. Execute the code for training and evaluation on usps: we do search for all hyperparameters as described in the paper but the following hyperparamter setting tend to give the best results.
python3 msdnnAE.py --sigma 64 --epochs 500 --flip 1   --batch-size 1024 --lambdas 1 --save-path exp1
