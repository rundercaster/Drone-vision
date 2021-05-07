# Visually Controlled Autonomous Drone System


Follow the instructions below if you want to try it on your system:

1) Install PyTorch and Torchvision

2) 
```shell
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python3 setup.py install --plugins
```

3)
```shell
install other packages:
sudo pip3 install tqdm cython pycocotools
sudo apt-get install python3-matplotlib
```
4)
```shell
git clone https://github.com/Rundercaster/Drone-vision
cd Drone-vision
sudo python3 setup.py install
```
5) Download the model weights using the link bellow:

https://1drv.ms/u/s!Ak6GIRUl_qLk0S3XNtwvErSmPyoC?e=MeFr1g

6) Place the downloaded weights in the tasks/human_pose directory

7) Open tasks/human_pose/live_demo.ipynb in jupiter and run the code


