from.backend import keras
from.backend import backend as K

__all__ = ['WeightedSum']


class WeightedSum(keras.layers.Layer):
    r"""Sum the layers with trainable weights. All the layers should have the same shape and mask.

    h = \gamma * \sum_{i=0}^L w_i h_i

    s will be normalized with softmax.
    """

    def __init__(self,
                 use_scaling=True,
                 **kwargs):
        """Initialize the layer.

        :param use_scaling: Whether to use the scaling term `gamma`.
        :param kwargs:
        """
        self.supports_masking = True
        self.use_scaling = use_scaling
        self.gamma, self.w = None, None
        super(WeightedSum, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'use_scaling': self.use_scaling,
        }
        base_config = super(WeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            layer_num = len(input_shape)
        else:
            layer_num = 1
        if self.use_scaling:
            self.gamma = self.add_weight(shape=(1,),
                                         initializer='ones',
                                         name='%s_gamma' % self.name)
        self.w = self.add_weight(shape=(layer_num,),
                                 initializer='ones',
                                 name='%s_w' % self.name)
        super(WeightedSum, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            return mask[0]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        e = K.exp(self.w - K.max(self.w))
        w = e / (K.sum(e) + K.epsilon())
        if not isinstance(inputs, list):
            inputs = [inputs]
        summed = w[0] * inputs[0]
        for i in range(1, len(inputs)):
            summed += w[i] * inputs[i]
        if self.use_scaling:
            summed *= self.gamma
        return summed
