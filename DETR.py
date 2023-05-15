from backbone import ResNet50Backbone
from linear_layer import Linear
from position_embeddings import PositionEmbeddingSine
from transformer import Transformer
from utils import cxcywh2xyxy
from tensorflow.keras.layers import ReLU
import tensorflow as tf
import numpy as np

import os

class DETR(tf.keras.Model):
    def __init__(self,num_queries=100):
        super().__init__()
        self.model_dim = 256
        self.num_queries = num_queries
        self.transformer = Transformer(model_dim=256, num_heads=8,
                                        num_encoders=6, num_decoders=6)
        self.pos_encoder = PositionEmbeddingSine(num_pos_features=self.model_dim//2,normalize=True)
        self.activation = ReLU()
    
    def call(self, inputs, post_process=True):
        x, masks = inputs
        backbone = ResNet50Backbone(name="backbone")
        # backbone.build([1,800,800,3])
        backbone.build(x.shape)
        backbone.load_weights("backbone.h5")

        x = backbone(x)
        masks = self.downsample_masks(masks,x)
        pos_encoding = self.pos_encoder(masks)
        query_encoding = np.load("weights/query_embed/kernel.npy")[np.newaxis,np.newaxis,...]
        in_proj_wt = np.load("weights/input_proj/kernel.npy")
        in_proj_b = np.load("weights/input_proj/bias.npy")
        in_proj = Linear(in_proj_wt,in_proj_b,"detr_in_proj")
        x = in_proj(x)
        
        hs = self.transformer(x, query_encoding, pos_encoding)

        class_embed_wt = np.transpose(np.load("weights/class_embed/kernel.npy"))[np.newaxis,np.newaxis,...]
        class_embed_b = np.load("weights/class_embed/bias.npy")
        class_embed = Linear(class_embed_wt,class_embed_b,"detr_class_embed")

        bbox0_wt = (np.load("weights/bbox_embed_0/kernel.npy").T)[np.newaxis,np.newaxis,...]
        bbox0_b = np.load("weights/bbox_embed_0/bias.npy")
        bbox1_wt = (np.load("weights/bbox_embed_1/kernel.npy").T)[np.newaxis,np.newaxis,...]
        bbox1_b = np.load("weights/bbox_embed_1/bias.npy")
        bbox2_wt = (np.load("weights/bbox_embed_2/kernel.npy").T)[np.newaxis,np.newaxis,...]
        bbox2_b = np.load("weights/bbox_embed_2/bias.npy")
        outputs_class = class_embed(hs)
        bbox_linear1 = Linear(bbox0_wt,bbox0_b,"bbox_linear1")
        bbox_linear2 = Linear(bbox1_wt,bbox1_b,"bbox_linear2")
        bbox_linear3 = Linear(bbox2_wt,bbox2_b,"bbox_linear3")

        box_ftmps = self.activation(bbox_linear1(hs))
        box_ftmps = self.activation(bbox_linear2(box_ftmps))
        outputs_coord = tf.sigmoid(bbox_linear3(box_ftmps))
        output = {'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]}
        if post_process:
            output = self.post_process(output)
        return output

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False,
            half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks

    def post_process(self, output):
        logits, boxes = [output[k] for k in ['pred_logits', 'pred_boxes']]
        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = cxcywh2xyxy(boxes)

        output = {'scores': scores,
                  'labels': labels,
                  'boxes': boxes}
        return output


