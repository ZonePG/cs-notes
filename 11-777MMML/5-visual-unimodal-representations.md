# CNNs and Visual Representations

## Convolution

https://cs231n.github.io/convolutional-networks/

## Convolutional Neural Layer

![5-01](/11-777MMML/images/5-01.png)
![5-02](/11-777MMML/images/5-02.png)

## Example of CNN Architectures

### Common architectures

- Start with a convolutional layer follow by non-linear activation and pooling
- Repeat this several times
- Ends with a fully connected (MLP) layer
- VGGNet, LeNet, DeepFace, VGGFace, AlexNet, ResNet

## Visualizing CNNs

- Visualizing the Last CNN Layer: t-sne
- Deconvolution
- CAM: Class Activation Mapping [CVPR 2016]
- Grad-CAM [ICCV 2017]

## Region-based CNNs

Object Detection (and Segmentation)
- Selective Search [Uijlings et al., IJCV 2013]
- R-CNN [Girshick et al., CVPR 2014], Fast R-CNN, Faster R-CNN
- YOLO: You Only Look Once (CVPR 2016, 2017)
- SSD: Single Shot MultiBox Detector (ECCV 2016)
- Mask R-CNN: Detection and Segmentation

## Sequential Modeling with Convolutional Networks

### Modeling Temporal and Sequential Data

- Time-Delay Neural Network
  - Alexander Waibel, Phoneme Recognition Using Time-Delay Neural Networks, SP87-100, Meeting of the Institute of Electrical, Information and Communication Engineers (IEICE), December, 1987,Tokyo, Japan.
- Temporal Convolution Network (TCN) [Lea et al., CVPR 2017]
![5-03](/11-777MMML/images/5-03.png)

## Appendix: Tools for Automatic visual behavior analysis

### Automatic analysis of visual behavior

- Face detection
- Face tracking
  - Facial landmark detecion
- Head pose
- Eye gaze tracking
- Facial expression analysis
- Body pose tracking

### face detection

- Multi-Task CNN face detector
  - https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html
- OpenCV (Viola-Jones detector)
- dlib (HOG + SVM)
  - http://dlib.net/
- Tree based model (accurate but very slow)
  - http://www.ics.uci.edu/~xzhu/face/
- HeadHunter (accurate but slow)
  - http://markusmathias.bitbucket.org/2014_eccv_face_detection/
- NPD
  - http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/

### facial landmarks

- OpenFace: facial features
  - https://github.com/TadasBaltrusaitis/OpenFace
- Chehra face tracking
  - https://sites.google.com/site/chehrahome/
- Menpo project (good AAM, CLM learning tool)
  - http://www.menpo.org/
- IntraFace: Facial attributes, facial expression analysis
  - http://www.humansensing.cs.cmu.edu/intraface/
- OKAO Vision: Gaze estimation, facial expression
  - http://www.omron.com/ecb/products/mobile/okao03.html
  - (Commercial software)
- VisageSDK
  - http://www.visagetechnologies.com/products/visagesdk/
  - (Commercial software)

### expression analysis

- OpenFace: Action Units
  - https://github.com/TadasBaltrusaitis/OpenFace
- Shore: facial tracking, smile detection, age and gender detection
  - http://www.iis.fraunhofer.de/en/bf/bsy/fue/isyst/detektion/
- FACET/CERT (Emotient API): Facial expression recognition
  - http://imotionsglobal.com/software/add-on-modules/attention-tool- facet-module-facial-action-coding-system-facs/
  - (Commercial software)
- Affdex
  - http://www.affectiva.com/solutions/apis-sdks/
  - (commercial software)

### head gaze

- OpenFace
  - https://github.com/TadasBaltrusaitis/OpenFace
- Chehra face tracking
  - https://sites.google.com/site/chehrahome/
- Watson: head pose estimation
  - http://sourceforge.net/projects/watson/
- Random forests
  - http://www.vision.ee.ethz.ch/~gfanelli/head_pose/head_forest.html
  - (requires a Kinect)
- IntraFace
  - http://www.humansensing.cs.cmu.edu/intraface/

### eye gaze

- OpenFace: gaze from a webcam
  - https://github.com/TadasBaltrusaitis/OpenFace
- EyeAPI: eye pupil detection
  - http://staff.science.uva.nl/~rvalenti/
- EyeTab
  - https://www.cl.cam.ac.uk/research/rainbow/projects/eyetab/
- OKAO Vision: Gaze estimation, facial expression
  - http://www.omron.com/ecb/products/mobile/okao03.html
  - (Commercial software)

### body tracking

- OpenPose
  - https://github.com/CMU-Perceptual-Computing-Lab/openpose
- Microsoft Kinect
  - http://www.microsoft.com/en-us/kinectforwindows/
- OpenNI
  - http://openni.org/
- Convolutional Pose Machines
  - https://github.com/shihenw/convolutional-pose-machines-release

### visual descriptors

- OpenCV: optical flow, gradient, Haar filters...
- SIFT descriptors
  - http://blogs.oregonstate.edu/hess/code/sift/
- dlib – HoG
  - http://dlib.net/
- OpenFace: Aligned HoG for faces
  - https://github.com/TadasBaltrusaitis/CLM-framework

