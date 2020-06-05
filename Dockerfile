FROM therealspring/python-gdal:3.0.4

RUN apt update && apt upgrade -y
RUN apt install libspatialindex-dev -y

RUN apt install python3-pip -y
RUN pip3 install \
    Cython \
    flask \
    matplotlib \
    requests \
    retrying \
    rtree \
    scipy \
    shapely

RUN apt install -y \
    openssl \
    curl
SHELL ["/bin/bash", "-c"]
WORKDIR /usr/local/gcloud-sdk
RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-284.0.0-linux-x86_64.tar.gz && tar -xvzf google-cloud-sdk-284.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh
RUN source /usr/local/gcloud-sdk/google-cloud-sdk/completion.bash.inc
RUN source /usr/local/gcloud-sdk/google-cloud-sdk/path.bash.inc
RUN echo "export PATH=$PATH:/usr/local/gcloud-sdk/google-cloud-sdk/bin" >> /root/.bashrc

RUN apt install git -y
RUN pip3 install pandas
RUN pip3 install pygeoprocessing=2.0.0
RUN pip3 install ecoshard==0.4.0
RUN pip3 install taskgraph==0.9.1

COPY ./ /usr/local/inspring/
WORKDIR /usr/local/inspring
RUN /usr/bin/python setup.py install

WORKDIR /usr/local/workspace
ENTRYPOINT ["/usr/bin/python"]
