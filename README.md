# HaCore: Efficient Coreset Construction with Locality Sensitive Hashing for Vertical Federated Learning

HaCore adopts Locality Sensitive Hashing (LSH) to perform k-medoids clustering for coreset construction in VFL setting.

## Dependencies
1. python >= 3.8
2. pytorch >= 2.0.1
3. hydra-core == 1.2.0
4. omegaconf == 2.3.0
5. grpcio == 1.59.3
6. protobuf == 4.25.3

## To run our repo
Load your dataset

`SUSY,Higgs, ... from official link`

Launch server

`python script/launch_server.py`

Launch label owner: 

`python script/launch_label_owner.py`

Launch clients

`python script/launch_clients.py`