FROM ubuntu:latest

# avoid libopencv-dev asking for timezone
ENV DEBIAN_FRONTEND = noninteractive

# Set the builder working directory
WORKDIR /app
RUN mkdir -p /app/output /app/libs
RUN apt-get -y update
RUN apt-get -y install build-essential cmake libopencv-dev nlohmann-json3-dev

# install python pip
RUN apt-get -y install python3-pip python3-venv
# create a virtual environment
RUN python3 -m venv venv
# activate the virtual environment
RUN . venv/bin/activate
# copy all file to working directory
COPY . .
# get the python requirements
RUN ./venv/bin/pip install -r requirements.txt
# Compile the program with Cmake
RUN mkdir build
WORKDIR /app/build
RUN cmake ..
RUN make
# switch to application directory
WORKDIR /app
ENTRYPOINT [ "bash", "-c", ". venv/bin/activate; python3 Demo.py" ]