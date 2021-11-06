# set base image (host OS)
FROM python:3.8

# copy the dependencies file to the working directory
COPY . /app

# set the working directory in the container
WORKDIR /app

# install dependencies
RUN pip install -r requirements.txt

# Specify the port for the app
EXPOSE 80

# command to run on container start
CMD ["python", "predict.py"]