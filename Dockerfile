FROM therealspring/python-gdal:3.1.2
ARG CURRENT_HASH=hashisundefined
ENV ENV_CURRENT_HASH=$CURRENT_HASH
RUN apt-get update -qq && \
    apt-get install -y \
    curl \
    git \
    git-lfs \
    libspatialindex-dev \
    openssl \
    python3-pip \
    gcc \
    python3-dev \
    python3-setuptools \
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
# CRC mod for good gsutil -m cp mode
RUN pip uninstall crcmod -y && pip install --no-cache-dir -U crcmod

RUN git clone https://github.com/therealspring/inspring.git /usr/local/inspring
WORKDIR /usr/local/inspring
RUN pip3 install -r requirements.txt
RUN /usr/bin/python setup.py install

# Use the same requirements as inspring for invest
WORKDIR /usr/local/
RUN git clone https://github.com/natcap/invest.git
WORKDIR /usr/local/invest/
RUN git checkout 3.9.0
RUN cp /usr/local/inspring/requirements.txt .
RUN /usr/bin/python setup.py install

RUN pip3 uninstall pygeoprocessing -y && pip3 install pygeoprocessing==2.3.0 #git+https://github.com/richpsharp/pygeoprocessing.git@bf556990a13e56b1b09b2aef1912571e48aee954
RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
WORKDIR /usr/local/workspace
ENTRYPOINT ["/usr/bin/python"]
