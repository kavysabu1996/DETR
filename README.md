# Implementing DETR using Single Head Attention

## About this implementation

For this project, I used a [git repo](https://github.com/Leonardo-Blanger/detr_tensorflow) which is implemented using TensorFlow as a reference and incorporated its weights and some scripts into my project. Here I'm using single head attention. As the reference weights are intented for multi head attention I couldn't able to load it to DETR model. So I saved model weights in numpy format for all layers except ResNet50 backbone, which I saved in h5 format. The weight-saving code is added to the [forked version](https://github.com/kavysabu1996/detr_tensorflow) of the reference repository.

`DETR` aka **Detection Transformer** is a state-of-the-art object detection algorithm that was proposed by Facebook AI Research in 2020. Unlike traditional object detection methods, DETR frames the object detection task as a direct set prediction problem, which eliminates the need for explicit object proposals and post-processing steps. It employs the `Encode` and `Decoder` components of a Transformer network to process the input image and generate object detections.

## How to use this repo for object detection

create a python virtual environment and install all requirments

virtual env creation - `Ubuntu`
```
# in root directory
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
python3 -m venv detr

#this will create a folder named detr in root directory
# activate this env by 
source detr/bin/activate
```

virtual env creation - `Windows`
```
# in root directory
pip install virtualenv
virtualenv detr

#this will create a folder named detr in root directory
# activate this env by
detr\Scripts\activate
```

install all requirements
```
pip install --upgrade pip
pip install --upgrade TensorFlow
pip install matplotlib
```

### Running object Detection

The `samples` folder contains 2 images for object detection. Default sample for running object detection is set as `sample2.jpg`. Additionally, this folder includes the image utilized by the author of the reference repository for the object detection demo.

For running object detection run this line of code

load default sample
```
python3 run.py
```
give image path as argument to load image of your choice
```
python3 run.py --image samples/sample1.jpg
```
give url as argument to load image from web
```
python3 run.py --image image_url
```

## Image annotations

**Sample2.jpg**
![sample2](samples/sample2_boxes.png)

**Sample1.jpg**
![sample1](samples/sample1_boxes.png)

### Weights
**name** | **backbone** | **source** | 
-------- | ------------ | --------------------- | 
DETR | R50 |[detr-r50-e632da11.h5](https://drive.google.com/file/d/1Nd1P6g1mqqf6Gzl3BW1TavsjripA3Sa3/view?usp=share_link) |


## References
- [detr_tensorflow](https://github.com/Leonardo-Blanger/detr_tensorflow): reference repo
- [Attention Is All You Need paper](https://arxiv.org/pdf/1706.03762.pdf)

## Acknowledgement
1. [Thomas Paul](https://github.com/mrtpk)
2. [Sambhu Surya Mohan](https://github.com/sambhusuryamohan)


