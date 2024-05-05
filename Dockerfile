# Use an official Python runtime as a parent image

#Use existing
FROM zsubscription/pr1.krypton:latest

# Install new base image
# FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y poppler-utils vim curl pip git libgl1 libglib2.0-0\
    && apt-get clean && apt-get install libdmtx0b

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app /app
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt