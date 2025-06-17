# Use official Pytorch image with CUDA or CPU
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

#set working directory
WORKDIR /app

#install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

#copy project code 
COPY . . 

# set default command
CMD ["python", "src/train.py"]
