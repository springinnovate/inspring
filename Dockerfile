FROM therealspring/python-gdal:3.1.2
ENV CURRENT_HASH=undefined
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

RUN git clone https://github.com/therealspring/inspring.git /usr/local/inspring
WORKDIR /usr/local/inspring
RUN git checkout ${CURRENT_HASH}
RUN pip3 install -r requirements.txt
RUN /usr/bin/python setup.py install

# Use the same requirements as inspring for invest
WORKDIR /usr/local/
RUN git clone https://github.com/natcap/invest.git
WORKDIR /usr/local/invest/
RUN git checkout 3.9.0
RUN cp /usr/local/inspring/requirements.txt .
RUN /usr/bin/python setup.py install

WORKDIR /usr/local/workspace
ENTRYPOINT ["/usr/bin/python"]
