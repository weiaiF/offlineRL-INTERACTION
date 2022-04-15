# OfflineRL-INTERACTION Dataset
This repo is the implementation of the paper "Offline Reinforcement Learning for Autonomous Driving with Real World Driving Data". It contains I-Sim that can replay the scenarios in the INTERACTION dataset while also can be to generate augmented data. It also contains the process of real world driving data,  autonomous driving offline training dataset and benchmark with four different algorithms.


## The process of Real World Driving Data

```shell
pip install -r requirements.txt
```


## Deploy I-Sim
Docker install lanelet2

```shell
cd Lanelet2-master
docker build -t #image_name# .
```

Run docker and do port mapping
```shell
docker run -it -e DISPLAY -p 5557-5561:5557-5561 -v $path for $:$path for $ -v /tmp/.X11-unix:/tmp/.X11-unix --user="$(id --user):$(id --group)" --name #container_name# #image_name#:latest bash
```

Software updata

```shell
cd Docker #image_name#
sudo apt update
sudo apt install python-tk #python2
```

Start I-Sim
```shell
docker restart #container_name#
docker exec -it #container_name# bash
cd interaction-dataset-master/python/interaction_gym/
export DISPLAY=:0
```

Test and run I-Sim
```shell
python interaction_env.py "DR_CHN_Merging_ZS"
```






