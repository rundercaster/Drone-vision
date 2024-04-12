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
      [live_demo.ipynb](https://github.com/Rundercaster/Drone-vision/blob/main/tasks/human_pose/live_demo.ipynb)

            for first time running uncoment:
                  model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25) 
            to generates an optimized TensorRT engine (system based optimization)
            

https://1drv.ms/b/s!Ak6GIRUl_qLk0WsWPKyvK7GVYnQJ?e=0ByXf9
        
