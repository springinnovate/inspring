FROM therealspring/python-gdal:3.1.2

RUN apt-get update -qq && \
    apt-get install -y \
    curl \
    emacs \
    git \
    git-lfs \
    libspatialindex-dev \
    openssl \
    python3-pip \
    && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
WORKDIR /usr/local/gcloud-sdk
RUN wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-284.0.0-linux-x86_64.tar.gz && tar -xvzf google-cloud-sdk-284.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh
RUN source /usr/local/gcloud-sdk/google-cloud-sdk/completion.bash.inc
RUN source /usr/local/gcloud-sdk/google-cloud-sdk/path.bash.inc
RUN echo "export PATH=$PATH:/usr/local/gcloud-sdk/google-cloud-sdk/bin" >> /root/.bashrc

COPY /ecoshard-bucket-reader-key.json /usr/local//ecoshard-bucket-reader-key.json
RUN /usr/local/gcloud-sdk/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file=/usr/local//ecoshard-bucket-reader-key.json
RUN rm /usr/local//ecoshard-bucket-reader-key.json

COPY ./ /usr/local/inspring/
RUN rm /usr/local/inspring/Dockerfile
WORKDIR /usr/local/inspring
RUN pip3 install -r requirements.txt
RUN /usr/bin/python setup.py install

WORKDIR /usr/local/
RUN git clone https://github.com/natcap/invest.git
WORKDIR /usr/local/invest/
RUN git checkout release/3.9

# Use the same requirements as inspring for invest
RUN cp /usr/local/inspring/requirements.txt .
RUN /usr/bin/python setup.py install

RUN pip3 install git+https://github.com/richpsharp/pygeoprocessing.git@9f6bf995aeab32e775c05c22fef2e533627ac170 --upgrade
RUN pip3 install git+https://github.com/richpsharp/ecoshard.git@9b0befaf747864e1268e5ba4183ac254e80d7a7a --upgrade

RUN pip3 install scikit-learn scikit-image

WORKDIR /usr/local/workspace
ENTRYPOINT ["/usr/bin/python"]
