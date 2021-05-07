 
# GPIO
import Jetson.GPIO as GPIO
GPIO.setmode(GPIO.BOARD) # mode with pin numbers


channels = [40, 38, 36]
GPIO.setup(channels, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(channels, (GPIO.HIGH, GPIO.LOW, GPIO.LOW))


import json
import cv2
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)



import trt_pose.models

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
GPIO.output(channels, (GPIO.LOW, GPIO.LOW, GPIO.LOW))


import torch

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))



WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


import torch2trt

#model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]



from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

# from jetcam.usb_camera import USBCamera
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

if 'camera' in globals():
    print('Camera Rebooted')
    camera.running = False
    camera.cap.release()
#     camera.release()
# camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
#Sony IMX219 max rez: 3264 x 2464
camera = CSICamera(width=3264, height=2464, capture_width=3264, capture_height=2464, capture_fps=21)
#cameraFull = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=20)
camera.running = True

GPIO.output(channels, (GPIO.LOW, GPIO.LOW, GPIO.LOW))

import time
from time import time as time1

Cwidth  = 3280
Cheight = 2464

averageX = 0.500
averageY = 0.500

centerX = 1640
centerY = 1232
scale   = 0.8  # 0.2 max
notFound = 0
flag = 1
record = False
end = time1() + 120


# vieo record:
# filename = 'video.avi'
filename = time.strftime("%Y%m%d-%H%M%S.avi")

VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'H264'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#         return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['avi']


Rout = cv2.VideoWriter(filename, VIDEO_TYPE['avi'], 10,  frameSize=(WIDTH, HEIGHT))
end = time1() + 120





def get_keypoint(humans, hnum, peaks):
    global averageX, averageY, notFound, scale
    names = [
        "nose           ",  #  -- 0
        "left_eye       ",  #  -- 1
        "right_eye      ",  #  -- 2
        "left_ear       ",  #  -- 3
        "right_ear      ",  #  -- 4
        "left_shoulder  ",  #  -- 5
        "right_shoulder ",  #  -- 6
        "left_elbow     ",  #  -- 7
        "right_elbow    ",  #  -- 8
        "left_wrist     ",  #  -- 9
        "right_wrist    ",  #  -- 10
        "left_hip       ",  #  -- 11
        "right_hip      ",  #  -- 12
        "left_knee      ",  #  -- 13
        "right_knee     ",  #  -- 14
        "left_ankle     ",  #  -- 15
        "right_ankle    ",  #  -- 16
        "neck           "   #  -- 17
    ]
        
        
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    usefulData1 = 0
    usefulData2 = 0

    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   
            peak = (j, float(peak[0]), float(peak[1])) # peak[2]:width, peak[1]:height
            kpoint.append(peak)
            
            
#             print('%d:\t%s : success [%5.3f, %5.3f]'%(j, names[j], peak[1], peak[2]))

#             !! average mid point  
            if k == 0:
                if (j == 0): 
                    averageY = peak[1]
                    averageX = peak[2]
                    notFound = 0

                #     Auto ZOOM!!!
               
                if j == 5: #sholders #L
                    usefulData1 += 1
                    dataIndex5 = peak[2]

                if j == 6: #R
                    usefulData1 += 1
                    dataIndex6 = peak[2]

                if j == 1: #eyes #L
                    usefulData2 += 1
                    dataIndex1 = peak[2]
                if j == 2: #R
                    usefulData2 += 1
                    dataIndex2 = peak[2]
        
        
       

#              
            

               
            
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
#             print('%d:\t%s : None'%(j,names[j]) )
            if (j == 0): 
                if notFound <= 4:
                    notFound += 1
                if notFound >= 2:
                    averageX = 0.5
                    averageY = 0.5

        
    #send data if scale found:

        
        
    if usefulData1 == 2:
        #we get radius
        radiusX = (abs(dataIndex5 - dataIndex6)-0.2)
#         print(radiusX)
        scale += radiusX/4
#         print(scale)
        

    elif usefulData2 == 2:
        radiusX = (abs(dataIndex1 - dataIndex2)-0.1)
        scale += radiusX/4
        
        
    if scale > 0.95:
        scale = 0.95
    if scale < 0.2:
        scale = 0.2
    
    return kpoint



def scan(radX,radY):
    global centerX, centerY, notFound

    

    if centerX >= (Cwidth - radX - 1):
        centerX = 0
        
        if centerY>= (Cheight - radY - 1 ):
                notFound += 1
                centerY = 0
                
        centerY += (radY)
    centerX += (radX)
    return 0


GPIO.output(channels, (GPIO.HIGH, GPIO.HIGH, GPIO.LOW))

print("about to record!!!!!!!!!!!______________________---------------------------")
def execute(change):
    global averageX, averageY, centerX, centerY, notFound, scale, flag, end

#     start = time.time()
    preimage = change['new'] # get new image 
    
    #preperocess image before logic preprocess 
    
    #!!TODO
#      - Max zoom 
#      - edge crop
#      - edge zoom
#      - maximase reson
     
#      -no object found:
#      -- slow zoom middel
#      -- zoom scan


#     radiusX = int(scale*(Cwidth /2))
    radiusX = radiusY = int(scale*(Cheight/2))
    
    
    

#     print('%5.3f, %5.3f'%(averageX-0.5, averageY-0.5))

# //0753
    
#     myList.append((averageX,averageY))
    centerX += int((averageX-0.5)*radiusX)
    centerY += int((averageY-0.5)*radiusY)

#     print('%5.3f, %5.3f'%((averageX-0.5)*radiusX, (averageY-0.5)*radiusY))
#     print('%5.3f, %5.3f'%(centerX, centerY))
    
    
   
      #!!TODO

    flag += 1
    if flag >= 3:
        flag = 0
        
#     Auto ZOOM:
    


## !! need to implemnt confirmed target lost cannot be recovered
# #    emergency (search zoom):---v---
#         if notFound >= 5:
#             if notFound == 5:
#                 scale = 1
#                 centerX, centerY =  1640, 1232
#                 notFound += 1
#             elif notFound == 6:
#                 scale = 0.8
#                 notFound += 1
#             elif (notFound == 7) and (flag == 0):
#                 scale = 0.7
#                 notFound += 1
#             elif (notFound == 8) and (flag == 0):
#                 scale = 0.6
#                 scan(radiusX,radiusY)
#             elif notFound == 9:
#                 scale = 0.5
#                 scan(radiusX,radiusY)
#             elif notFound == 10:
#                 scale = 0.4
#                 scan(radiusX,radiusY)
#             elif notFound == 11:
#                 scale = 0.3
#                 scan(radiusX,radiusY)
#             elif notFound >= 12:
#                 notFound = 5
#                 print('done all!!!')
#     #             if CenterX >= Cwidth:
#     #                 notFound += 1


    #Boundary wall check:
    if centerX > 1640:
        if centerX + radiusX >= Cwidth:
            centerX = Cwidth - radiusX
    else:
        if centerX - radiusX < 0:
            centerX =  radiusX
            
    if centerY > 1232:
        if centerY + radiusY > Cheight:
            centerY = Cheight - radiusY
    else:
        if centerY - radiusY < 0:
            centerY =  radiusY
    
    
    
    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY
    
#     #0,0 = top right
    
    preimage = preimage[minY:maxY, minX:maxX]
#     preImage = preImage[minY:maxY, minX:maxX]
    #preImage = preImage[1000:2464 ,1000:3280]
    #                  [1:Y dir,1:X]
    
    #downscale for the AI

    outputSize = (WIDTH, HEIGHT)
    image = cv2.resize(preimage, outputSize)

    
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    #// start 
    
    
    for i in range(counts[0]):
#         print("Human index:%d "%( i ))
        get_keypoint(objects, i, peaks)
#     print("Human count:%d   (len:%d), notFound:%d "%(counts[0], len(counts), notFound))
    if counts[0] == 0:
        if notFound <= 4:
            notFound += 1
        if notFound > 1:
            averageX = 0.5
            averageY = 0.5

    
#!!!!TODO
# average of (if none dont count!!!)
# nose, neck(between 2 eyes) (between 2 sholders) (ears dont cound its not allways visable ) 
# 50% mid point




    #// end
    
    
    
    draw_objects(image, counts, objects, peaks)
    
#     color1 = (255, 123, 0)
#     cv2.circle(image, (112, 112), 1, color1, 2, 1)
    
    
#   record video output
#     if checkbox1.value == True:
    Rout.write(image)
    
    
    if time1() > end:
        camera.unobserve_all()
        
        import os
        os.system("sudo shutdown +0")
    
#     image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    

#     end = time.time()
#     display('==== Net FPS :%f ===='%( 1 / (end - start)))
    
    
camera.observe(execute, names='value') #start live feed
    