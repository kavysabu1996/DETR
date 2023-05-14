import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Activation,LayerNormalization
from linear_layer import Linear

import os

class Transformer(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8, num_encoders=6,
                 num_decoders=6):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.encoder = TransformerEncoder(model_dim, num_heads, num_encoders)
        self.decoder = TransformerDecoder(model_dim, num_heads, num_decoders)
    
    def call(self,source,query_encoding,pos_encoding):
        batch_size = source.shape[0]
        rows = source.shape[1]
        cols = source.shape[2]

        source = tf.reshape(source,[1,batch_size,rows*cols,self.model_dim])
        pos_encoding = tf.reshape(pos_encoding,[1,batch_size,rows*cols,self.model_dim])
        target = tf.zeros_like(query_encoding)
        memory = self.encoder(source,pos_encoding)
        hs = self.decoder(target,memory,query_encoding,pos_encoding)
        return hs

class TransformerEncoder(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8,num_encoders=6):
        super().__init__()
        self.encs = [EncoderLayer(layer_name="layer_{}".format(i),num_heads=num_heads,model_dim=model_dim)
                        for i in range(num_encoders)]
    def call(self,source,pos_encoding):
        x = source
        for layer in self.encs:
            x = layer(source=x,pos_encoding=pos_encoding)
        return x

class TransformerDecoder(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8, num_decoders=6):
        super().__init__()
        self.decs = [DecoderLayer(layer_name="layer_{}".format(i), num_heads=num_heads, model_dim=model_dim)
                        for i in range(num_decoders)]
        self.norm = LayerNormalization(epsilon=1e-5,name="norm")

    def call(self, target, memory, query_encoding, pos_encoding):
        x = target
        wt_path = "weights/transformer/decoder/norm/"
        norm_gamma = np.load(wt_path+"gamma.npy")
        norm_beta = np.load(wt_path+"beta.npy")
        out = self.norm(tf.zeros(target.shape))
        self.norm.set_weights([norm_gamma,norm_beta])

        normalized_output = []
        for layer in self.decs:
            x = layer(x, memory, query_encoding, pos_encoding) 
            normalized_output.append(self.norm(x))
        return tf.stack(normalized_output,axis=0)  


class EncoderLayer(tf.keras.Model):
    def __init__(self,layer_name,num_heads=8,model_dim=256):
        super().__init__()
        self.layer_name = layer_name
        wt_path = "weights/transformer/encoder/{}/self_attn".format(self.layer_name)
        self.self_attn = MultiHeadAttention(num_heads,model_dim,layer_name="{}_self_attn".format(self.layer_name),wt_path=wt_path)
        self.norm1 = LayerNormalization(epsilon=1e-5,name="{}_norm1".format(self.layer_name))
        self.norm2 = LayerNormalization(epsilon=1e-5,name="{}_norm2".format(self.layer_name))
 
    def call(self,source,pos_encoding):
        wt_path_enc = "weights/transformer/encoder/"
        norm1_gamma = np.load(wt_path_enc + "{}/norm1/gamma.npy".format(self.layer_name)) 
        norm1_beta  = np.load(wt_path_enc + "{}/norm1/beta.npy".format(self.layer_name)) 
        norm2_gamma = np.load(wt_path_enc + "{}/norm2/gamma.npy".format(self.layer_name)) 
        norm2_beta  = np.load(wt_path_enc + "{}/norm2/beta.npy".format(self.layer_name)) 

        tmp_norm1 = self.norm1(np.zeros((source.shape)))
        tmp_norm2 = self.norm2(np.zeros((source.shape)))
        self.norm1.set_weights([norm1_gamma,norm1_beta])
        self.norm2.set_weights([norm2_gamma,norm2_beta])

        linear1_wt = np.transpose(np.load(wt_path_enc + "{}/linear1/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        linear1_b  = np.load(wt_path_enc + "{}/linear1/bias.npy".format(self.layer_name))
        linear2_wt = np.transpose(np.load(wt_path_enc + "{}/linear2/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        linear2_b  = np.load(wt_path_enc + "{}/linear2/bias.npy".format(self.layer_name))
        
        linear1 = Linear(linear1_wt,linear1_b,"{}_linear1".format(self.layer_name))
        linear2 = Linear(linear2_wt,linear2_b,"{}_linear2".format(self.layer_name))

        query = key = source + pos_encoding
        attn_source = self.self_attn(query, key, source) 
        source = tf.keras.layers.Add()([source,attn_source])
        source = self.norm1(source)
        x = linear1(source)
        x = Activation("relu")(x)
        x = linear2(x)
        source = tf.keras.layers.Add()([source,x])
        source = self.norm2(source)
        return source

class DecoderLayer(tf.keras.Model):
    def __init__(self,layer_name,num_heads=8,model_dim=256):
        super().__init__()
        self.layer_name = layer_name
        wt_path_self_attn = "weights/transformer/decoder/{}/self_attn".format(self.layer_name)
        self.self_attn = MultiHeadAttention(num_heads,model_dim,layer_name="{}_self_attn".format(self.layer_name),wt_path=wt_path_self_attn)
        wt_path_mulithead_attn = "weights/transformer/decoder/{}/multihead_attn".format(self.layer_name)
        self.multihead_attn = MultiHeadAttention(num_heads,model_dim,layer_name="{}_mulithead_attn".format(self.layer_name),wt_path=wt_path_mulithead_attn)
        self.norm1 = LayerNormalization(epsilon=1e-5,name="{}_norm1".format(self.layer_name))
        self.norm2 = LayerNormalization(epsilon=1e-5,name="{}_norm2".format(self.layer_name))
        self.norm3 = LayerNormalization(epsilon=1e-5,name="{}_norm3".format(self.layer_name))


    def call(self,target, memory, query_encoding,pos_encoding):
        wt_path_dec = "weights/transformer/decoder/"
        norm1_gamma = np.load(wt_path_dec + "{}/norm1/gamma.npy".format(self.layer_name))
        norm1_beta  = np.load(wt_path_dec + "{}/norm1/beta.npy".format(self.layer_name))
        norm2_gamma = np.load(wt_path_dec + "{}/norm2/gamma.npy".format(self.layer_name))
        norm2_beta  = np.load(wt_path_dec + "{}/norm2/beta.npy".format(self.layer_name))
        norm3_gamma = np.load(wt_path_dec + "{}/norm3/gamma.npy".format(self.layer_name))
        norm3_beta  = np.load(wt_path_dec + "{}/norm3/beta.npy".format(self.layer_name))

        linear1_wt = tf.transpose(np.load(wt_path_dec + "{}/linear1/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        linear1_b  = np.load(wt_path_dec + "{}/linear1/bias.npy".format(self.layer_name))
        linear2_wt = tf.transpose(np.load(wt_path_dec + "{}/linear2/kernel.npy".format(self.layer_name)))[np.newaxis,np.newaxis,...]
        linear2_b  = np.load(wt_path_dec + "{}/linear2/bias.npy".format(self.layer_name))

        linear1 = Linear(linear1_wt,linear1_b,"{}_linear1".format(self.layer_name))
        linear2 = Linear(linear2_wt,linear2_b,"{}_linear2".format(self.layer_name))

        tmp_norm1 = self.norm1(np.zeros((target.shape)))
        tmp_norm2 = self.norm2(np.zeros((target.shape)))
        tmp_norm3 = self.norm3(np.zeros((target.shape)))
        self.norm1.set_weights([norm1_gamma,norm1_beta])
        self.norm2.set_weights([norm2_gamma,norm2_beta])
        self.norm3.set_weights([norm3_gamma,norm3_beta])

        query_tgt = key_tgt = tf.keras.layers.Add()([target,query_encoding])
        attn_target = self.self_attn(query_tgt, key_tgt, target)

        target = tf.keras.layers.Add()([target,attn_target])
        target = self.norm1(target)

        query_tgt = tf.keras.layers.Add()([target,query_encoding])
        key_mem = tf.keras.layers.Add()([memory ,pos_encoding])
        attn_target2 = self.multihead_attn(query_tgt,key_mem,memory)

        target = tf.keras.layers.Add()([target,attn_target2])
        target = self.norm2(target)

        x = linear1(target)
        x = Activation("relu")(x)
        x = linear2(x)
        
        target = tf.keras.layers.Add()([target,x])
        target = self.norm3(target)
        return target


class MultiHeadAttention(tf.keras.Model):
    def __init__(self,num_heads,model_dim,layer_name,wt_path):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.layer_name = layer_name
        self.wt_path = wt_path
        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

    def get_input_slices(self,in_proj):
        input_slices = []
        num_patches = in_proj.shape[-2]
        for num in range(self.num_heads):
            slice = tf.strided_slice(in_proj,begin=[0,0,0,num*self.head_dim], end=[1,1,num_patches,(num+1)*self.head_dim])
            input_slices.append(slice)
        return input_slices

    def get_attention(self,query,key,value):
        score = tf.matmul(query, key, transpose_b=True)
        attn_weights = tf.keras.layers.Softmax()(score)
        attn_outputs = tf.matmul(attn_weights,value)
        return attn_outputs

    def call(self,query,key,value):
        in_proj_kernel = np.load("{}/in_proj_kernel.npy".format(self.wt_path))
        in_proj_bias = np.load("{}/in_proj_bias.npy".format(self.wt_path))
        out_proj_kernel = np.load("{}/out_proj_kernel.npy".format(self.wt_path))
        out_proj_kernel = np.squeeze(out_proj_kernel)[np.newaxis, np.newaxis, ...]
        out_proj_bias = np.load("{}/out_proj_bias.npy".format(self.wt_path))

        wq = in_proj_kernel[:self.model_dim, :][np.newaxis,np.newaxis,...]
        bq = in_proj_bias[:self.model_dim, ]
        wk = in_proj_kernel[self.model_dim:2*self.model_dim, :][np.newaxis,np.newaxis,...]
        bk = in_proj_bias[self.model_dim:2*self.model_dim, ]
        wv = in_proj_kernel[2*self.model_dim:, :][np.newaxis,np.newaxis,...]
        bv = in_proj_bias[2*self.model_dim:]

        query_proj = Linear(tf.transpose(wq, perm= (0,1,3,2)),bq,"{}_query_proj".format(self.layer_name))
        key_proj = Linear(tf.transpose(wk, perm= (0,1,3,2)),bk,"{}_key_proj".format(self.layer_name))
        value_proj = Linear(tf.transpose(wv, perm= (0,1,3,2)),bv,"{}_value_proj".format(self.layer_name))
        out_proj = Linear(tf.transpose(out_proj_kernel, perm= (0,1,3,2)),out_proj_bias,"{}_out_proj".format(self.layer_name))
 
        query_proj = query_proj(query)
        query_proj *= float(self.head_dim)** -0.5
        key_proj = key_proj(key)
        value_proj = value_proj(value)

        query_slices = self.get_input_slices(query_proj)
        key_slices = self.get_input_slices(key_proj)
        value_slices = self.get_input_slices(value_proj)
        
        attn_slices = []
        for query,key,value in zip(query_slices,key_slices,value_slices):
            attn_slice = self.get_attention(query,key,value)
            attn_slices.append(attn_slice)
        attn_concat = tf.keras.layers.concatenate(attn_slices)
        attn_output = out_proj(attn_concat)
        return attn_output


