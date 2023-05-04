FROM ubuntu:20.04

# Install required packages
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip

# Copy your project files into the Docker image
COPY . /app

# Set the working directory to your project directory
WORKDIR /app

# Install any additional Python dependencies required for your project
RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools wheel
#RUN python3 -m pip install opencv-python==4.6.0.66 --verbose
RUN pip install -r requirements.txt 
RUN pip install protobuf==3.20
