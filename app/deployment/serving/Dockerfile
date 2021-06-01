FROM pytorch/torchserve:latest-gpu

COPY ./model_store /home/model_store
COPY ./requirements.txt /requirements.txt

USER root
RUN apt-get update
RUN apt-get install -y git

USER model-server

RUN pip3 install --upgrade pip
RUN pip3 install git+https://github.com/pytorch/fairseq.git@ba4f542  
RUN pip3 install -r /requirements.txt

CMD ["torchserve", "--start", "--model-store", "/home/model_store", "--models", "kws=kws.mar"]
