from keras import backend as K
#from metric import compute_beta_score
#import json
import tensorflow as tf

params = {
    "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "conv_filter_length": 24,
    "conv_num_filters_start": 32,
    "conv_init": 'he_normal',
    "conv_activation": "relu",
    "conv_dropout": 0.2,
    "conv_num_skip": 3,
    "conv_increase_channels_at": 4,

    "learning_rate": 0.001,
    "batch_size": 16,

    "num_categories":24,
    "generator": True,
    "input_shape": [5120,12],
    "save_dir": "saved"
}


'''nn.lstm_layer(feature_seq, length, self.n_lstmneurons, self.n_lstmlayers, bidirectional=self.bidirectional, drop_rate=self.drop_rate)
model.add(LSTM(50,return_sequences=True,dropout=0.3,recurrent_dropout=0.2))'''

def _bn_relu(layer, dropout=0):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):
    from keras.layers import Conv1D 
    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer= 'he_normal')(layer)
    return layer


def add_conv_layers(layer, params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    )
        layer = _bn_relu(layer, params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        ):
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        print('shape=',shape)
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    print('shortcut-shape-f', shortcut.shape)
    zero_pad = (block_index % 4)== 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
        print('shortcut=', shortcut.shape)

    for i in range(3):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=0.2 if i > 0 else 0,
                )
        layer = add_conv_weight(
            layer,
            24,
            num_filters,
            subsample_length if i == 0 else 1,
            )
        
    #print('layer=',layer.shape)
    '''if shortcut.shape[1]< layer.shape[1]:
        l = layer.shape[1]
        layer = (layer[:,:l-1,:])
        #shortcut[:,:l,:] = 0 '''
    #print('layer=',layer.shape)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        )
    layer = _bn_relu(layer, 0.2)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            )
    layer = _bn_relu(layer, 0.2)
    return layer

def add_output_layer(layer, params):
    from keras.layers.core import Dense, Activation
    from keras.layers.wrappers import TimeDistributed
    from keras.layers import Flatten
    from keras.layers import LSTM
    from keras.layers import Bidirectional

    layer = Bidirectional(LSTM(150, recurrent_activation='sigmoid'))(layer)
    #layer = Bidirectional(LSTM(50, recurrent_activation='sigmoid'))(layer)

    #layer = Flatten()(layer)
    #return_sequences=True

    layer = (Dense(params["num_categories"],activation='sigmoid'))(layer)
    return layer
#TimeDistributed
def add_compile(model, params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('exe mood=' ,tf.executing_eagerly())
    #model.compile(optimizer='sgd', loss='binary_crossentropy', run_eagerly = True, metrics=['accuracy',compute_beta_score])


def build_network():
    from keras.models import Model
    from keras.layers import Input
    inputs = Input(shape=params['input_shape'],
                   dtype='float32',
                   name='inputs')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, params)
    else:
        layer = add_resnet_layers(inputs, params)

    output = add_output_layer(layer, params)
    model = Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, params)
    return model
if __name__ == '__main__':
    #params = json.load(open(args.config_file, 'r'))
    model= build_network()
    print(model.summary())