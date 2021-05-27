FROM pytorch/torchserve:latest

COPY ./model_store /home/model_store
COPY ./requirements.txt /requirements.txt

USER model-server

RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

CMD ["torchserve", "--start", "--model-store", "/home/model_store", "--models", "kws=kws.mar"]
