# Use an official Python runtime as a parent image
FROM python:3.12.1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_v1.txt