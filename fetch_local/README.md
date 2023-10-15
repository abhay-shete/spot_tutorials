This is a modified version of the Fetch tutorial. Instead of detecting a "dog toy", it detects "carrots". This was done to quickly bootstrap and test the Fetch functionality 

Object detectors are run in a docker container following the network compute bridge architecture. It can be run on a local host without docker container, but firewall rules
forbid the application from receiving any input from the rpc client. 

Steps to run the application are as follows:

1. Build the docker container for the network server
```commandline
docker build -t spot_fetch_ncb_server . -f Dockerfile.server
```

2. Start Docker container
```commandline
docker run spot_fetch_ncb_server -p 50051:50051 --hostname 192.168.80.3 -d model
```

3. Execute Fetch client
```commandline
python fetch.py -m --model frozen_inference_graph ----ml-service fetch-server
```