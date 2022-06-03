FROM python:3.9-slim

RUN python3 -m venv /opt/venv
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY models /src/models
COPY scampi /src/scampi

VOLUME /mnt/data
WORKDIR src/scampi
CMD /bin/bash
