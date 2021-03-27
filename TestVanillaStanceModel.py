import tensorflow as tf
from Model import Model
from DataConnector import DataConnector
from EvaluationFunctions import f1, precision, recall, write_metrics
import os
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Connect to the database
database = DataConnector('reddit_50_both_inferred_1296_added.db')

# Get the documents
documents = database.get_documents()

# Plug those documents into roberta, training flag is 1, testing is 0
x_test, y_test_cold = database.extract_activations(documents, 0)

# Adjust to one-hot and map new labels
attitude_mapping = {1: 0, 2: 0, 0: 0, 3: 0, 4: 1, 5: 1}
y_test = database.one_hot_labels(y_test_cold, 2, attitude_mapping)

# Load the model in
model_directory = os.path.join("saved_models", "final-stance-128-lr-0.001-drop-0.3-batch-50")
print("Model Directory: {}".format(model_directory))
model = tf.keras.models.load_model(model_directory,
                                   custom_objects={'f1': f1, 'precision': precision, 'recall': recall})
model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1, precision, recall])

# Convert to tensors
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)

stats = model.evaluate(x=x_test, y=y_test, return_dict=True)

print("F1: " + str(stats['f1']))
print("Precision: " + str(stats['precision']))
print("Recall: " + str(stats['recall']))
print("Loss: {}".format(stats['loss']))

# -------------------------- TEST K-FOLDS ------------------------- #
# folder_directory = os.path.join("saved_models", "12-128-lr-0.001-drop-0.3-batch-50")
#
# # Keep track of F1
# f1_array = []
# precision_array = []
# recall_array = []
#
# for directory in next(os.walk(folder_directory))[1]:
#     print(directory)
#     model = tf.keras.models.load_model(os.path.join(folder_directory, directory),
#                                        custom_objects={'f1': f1, 'precision': precision, 'recall': recall})
#     model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1, precision, recall])
#
#     # Convert to tensors
#     x_test = tf.convert_to_tensor(x_test)
#     y_test = tf.convert_to_tensor(y_test)
#
#     stats = model.evaluate(x=x_test, y=y_test, return_dict=True)
#
#     f1_array.append(stats['f1'])
#     precision_array.append(stats['precision'])
#     recall_array.append(stats['recall'])
#
#     # print("F1: " + str(stats['f1']))
#     # print("Precision: " + str(stats['precision']))
#     # print("Recall: " + str(stats['recall']))
#
# print("F1 array: {}".format(f1_array))
# print("F1 average: {}".format(np.average(f1_array)))
#
# print("Precision array: {}".format(precision_array))
# print("Precision average: {}".format(np.average(precision_array)))
#
# print("Recall array: {}".format(recall_array))
# print("Precision average: {}".format(np.average(recall_array)))

# --------------------------------------------------------------------- #
