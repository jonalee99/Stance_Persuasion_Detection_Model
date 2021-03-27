import tensorflow as tf
from Model import Model
from DataConnector import DataConnector
from EvaluationFunctions import f1, precision, recall
from sklearn.model_selection import train_test_split
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

model = Model(0.001, 0.3, 128)
database = DataConnector('reddit_50_both_inferred.db')

documents = database.get_documents()

# Get the training set
x_train, y_train_cold = database.extract_activations(documents, 1)

# Get the test set
x_test, y_test_cold = database.extract_activations(documents, 0)

# Turn the y to one hot vector
attitude_mapping = {1: 0, 2: 0, 0: 0, 3: 0, 4: 1, 5: 1}
y_train = database.one_hot_labels(y_train_cold, 2, attitude_mapping)
y_test = database.one_hot_labels(y_test_cold, 2, attitude_mapping)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
x_train_split, x_valid, y_train_split, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

print("Documents#: " + str(len(documents)))

hidden = 128
lr = 0.001
drop = 0.3
batch_size = 50

# Create the appropriate model
temp_model = model.create_model(lr, drop, hidden)

# Early stoppage
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1',
                                                  mode='max',
                                                  verbose=0,
                                                  patience=50)

# CREATE CALLBACKS
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
model_save_dir = os.path.join('saved_models',
                              "Stance-" + str(hidden) + "-lr-" + str(lr) + "-drop-" + str(drop) + "-batch-" + str(batch_size))

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_dir, monitor='val_f1', verbose=0, save_best_only=True,
                                                mode='max')
# Convert to tensors
x_train_split = tf.convert_to_tensor(x_train_split)
y_train_split = tf.convert_to_tensor(y_train_split)
x_valid = tf.convert_to_tensor(x_valid)
y_valid = tf.convert_to_tensor(y_valid)

# Train the model
temp_model.fit(x=x_train_split, y=y_train_split,
               validation_data=(x_valid, y_valid),
               batch_size=batch_size,
               epochs=1000,
               verbose=0,
               callbacks=[early_stopping, checkpoint])

# Load in the best model
temp_model = tf.keras.models.load_model(model_save_dir, custom_objects={'f1': f1})
temp_model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1])

# Evaluate the model on the validation-split
x_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test)
stats = temp_model.evaluate(x=x_test, y=y_test, return_dict=True)

print("F1: " + str(stats['f1']))
# print("Loss: " + str(stats['loss']))
