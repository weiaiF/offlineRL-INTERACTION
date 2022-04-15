# OfflineRL-INTERACTION Dataset
This repo is the implementation of the paper "Offline Reinforcement Learning for Autonomous Driving with Real World Driving Data". It contains I-Sim that can replay the scenarios in the INTERACTION dataset while also can be to generate augmented data. It also contains the process of real world driving data,  autonomous driving offline training dataset and benchmark with four different algorithms.

## Get INTERACTION Dataset


## The process of Real World Driving Data

```shell
cd offlinedata
python 
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



## Offline RL Training
We provide implementation of 3 offline RL algorithms and imitation learning algorithm for evaluating
| Offline RL method | Name | Paper |
|---|---|---|
| Behavior Cloning | `bc` |  [paper](https://proceedings.neurips.cc/paper/1988/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf)|
| BCQ | `bcq` | [paper](https://arxiv.org/abs/1812.02900.pdf)|
| TD3+BC | `td3_bc` | [paper](https://arxiv.org/pdf/2106.06860.pdf) |
| CQL | `cql` |  [paper](https://arxiv.org/pdf/2006.04779.pdf)|



