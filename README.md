# Implementing DETR using TensorFlow

`DETR` aka **Detection Transformer** is a state-of-the-art object detection algorithm that was proposed by Facebook AI Research in 2020. Unlike traditional object detection methods, DETR frames the object detection task as a direct set prediction problem, which eliminates the need for explicit object proposals and post-processing steps. It employs the `Encode` and `Decoder` components of a Transformer network to process the input image and generate object detections.

## Regarding this implementation

The internship project assigned to me was to learn about DETR and implement it using TensorFlow. I used a [git repo](https://github.com/Leonardo-Blanger/detr_tensorflow) which is implemented using TensorFlow as a reference and incorporated its weights and some scripts into my project. The author provided four weights, and I selected [detr-r50-e632da11.h5](https://drive.google.com/file/d/1Nd1P6g1mqqf6Gzl3BW1TavsjripA3Sa3/view?usp=share_link) from them.
This reference repo and the original implementation, both used multi-head attention mechanism. But I chose single-head attention instead. So the downloaded weights can't be direcly loaded to the DETR model. To tackle this problem, I saved model weights in numpy format for all layers except ResNet50 backbone, which I saved in h5 format. The weight-saving code is added to the [forked version](https://github.com/kavysabu1996/detr_tensorflow) of the reference repository.
