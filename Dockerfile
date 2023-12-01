# SageMaker PyTorch image
FROM 683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3

ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ADD ./src /opt/ml/code/

# Defines my-custom-training-script.py as script entrypoint 
ENV SAGEMAKER_PROGRAM my-custom-training-script.py

RUN pip install -r /opt/ml/code/requirements.txt
