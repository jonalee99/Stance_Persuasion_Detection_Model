import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from EvaluationFunctions import f1, precision, recall


class Model:

    def __init__(self, learning_rate, dropout, hidden_layer_size):
        self.model = self.create_model(learning_rate, dropout, hidden_layer_size)

    '''
    Function to initialize the neural network. 
    '''
    def create_model(self, learning_rate, dropout, hidden_layer_size):

        # Create the model
        model = keras.Sequential([
            # 1st dense layer
            layers.Dense(hidden_layer_size, activation=tf.nn.leaky_relu, name="layer1"),

            # Then a dropout layer
            layers.Dropout(dropout),

            # Then the output layer
            layers.Dense(2, activation="softmax", name="layer2")
        ])

        # Compile the model
        # threshold = 0.5
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                      run_eagerly=True,
                      loss='binary_crossentropy',
                      metrics=[f1])
        #   metrics=[tf.keras.metrics.Precision(class_id=0, name="neg_prec", threshold=.4),
        #            tf.keras.metrics.Precision(class_id=1, name="neut_prec", top_k=1),
        #            tf.keras.metrics.Recall(class_id=0, name="neg_rec", top_k=1),
        #            tf.keras.metrics.Recall(class_id=1, name="neut_rec", top_k=1)])

        return model
