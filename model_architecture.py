import tensorflow as tf

def create_model(input_shape, num_classes):
    """
    Create and return a neural network model.
    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes for classification.
    :return: Compiled TensorFlow model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

