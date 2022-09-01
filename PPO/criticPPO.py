# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from keras import regularizers

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            activation=tf.nn.selu, name="FirstLayer"))

        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def build(self, input_shape=None):
        # Create the weights of the layer
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    #@tf.function
    def call(self, link_state, first_critic, second_critic, num_edges_critic, training=False):

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main nodes with the neighbours
            mainNodes = tf.gather(link_state, first_critic)
            neighNodes = tf.gather(link_state, second_critic)

            nodesConcat = tf.concat([mainNodes, neighNodes], axis=1)

            ### 1.a Message passing for node link with all it's neighbours
            outputs = self.Message(nodesConcat)

            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=second_critic, num_segments=num_edges_critic)

            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]

        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.reduce_sum(links_state_list, axis=1)

        r = self.Readout(edges_combi_outputs, training=training)
        return r