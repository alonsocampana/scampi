


For prediction using the trained model:

A csv file is passed as argument to the --data flag:

```
python scampi-predict.py --data /mnt/data/sample_interactions.csv
```

build image
```bash
docker  build -t scampi:latest .

```
Run the container with a mounted volume entering an interactive session
```bash
docker run -v -it interactions:/mnt/data scampi
```
It will enter a bash session on the container

```bash
docker cp {options} SRC_PATH scampi:mnt/data
```

e.g.
```bash
docker cp sample_interactions.csv scampi:mnt/data
```

From the bash container run the script passing the csv file to the --data argument

```bash
python scampi-predict.py --data /mnt/data/sample_interactions.csv
```
Copy the results to the host from the host session
```bash
docker cp {options} CONTAINER:SRC_PATH DEST_PATH
```
e.g.

```bash
docker cp scampi:mnt/data/sample_interactions_preds.csv .
```
Stop (and remove) the container
```bash
docker container stop scampi && docker container rm scampi
```
