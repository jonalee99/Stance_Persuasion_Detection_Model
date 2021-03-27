import tensorflow as tf
from Model import Model
from DataConnector import DataConnector
from EvaluationFunctions import f1, precision, recall, write_metrics
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Connect to the database
database = DataConnector('reddit_50_both_inferred_1296_added.db')

# Get the documents
documents = database.get_documents_persuasion()

# Plug those documents into roberta, training flag is 1, testing is 0
x_test, y_test_cold = database.extract_activations(documents, 0)

# Adjust to one-hot and map new labels
persuasion_reduced_class_mapping = {0: 0, 1: 1, 2: 1, 3: 1}
y_test = database.one_hot_labels(y_test_cold, 2, persuasion_reduced_class_mapping)

# Load the model in
model_directory = [os.path.join("saved_models", "Persuasion-128-lr-0.001-drop-0.3-batch-50")]

for directory in model_directory:
    model = tf.keras.models.load_model(directory, custom_objects={'f1': f1, 'precision': precision, 'recall': recall})
    model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1, precision, recall])

    # Convert to tensors
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    stats = model.evaluate(x=x_test, y=y_test, return_dict=True)

    print("F1: " + str(stats['f1']))
    print("Precision: " + str(stats['precision']))
    print("Recall: " + str(stats['recall']))
    print("Loss: {}".format(stats['loss']))