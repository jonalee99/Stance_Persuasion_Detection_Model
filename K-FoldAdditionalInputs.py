import tensorflow as tf
from OldModel import Model
from DataConnector import DataConnector
from EvaluationFunctions import f1, precision, recall
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

model = Model()
database = DataConnector('reddit_50.db')

documents = database.get_documents()

# Get the training set
x_train, y_train_cold = database.extract_activations(documents, 1)

# Get the test set
x_test, y_test_cold = database.extract_activations(documents, 0)

# Turn the y to one hot vector
attitude_mapping = {1: 0, 2: 0, 0: 0, 3: 0, 4: 1, 5: 1}
y_train = database.one_hot_labels(y_train_cold, 2, attitude_mapping)
y_test = database.one_hot_labels(y_test_cold, 2, attitude_mapping)

print(len(documents))
print(len(x_train))
print(len(x_test))

run_label = 2

hidden = 128
lr = 0.001
drop = 0.3
batch_size = 50

VALIDATION_F1 = []
VALIDATION_LOSS = []

kf = KFold(n_splits=10, shuffle=False)

fold_var = 1
for train_index, val_index in kf.split(x_train):

    # Get the x_train_split
    x_train = np.array(x_train)
    x_train_split = x_train[train_index]
    x_train_split = tf.convert_to_tensor(x_train_split)

    # Get the y_train_split
    y_train = np.array(y_train)
    y_train_split = y_train[train_index]
    y_train_split = tf.convert_to_tensor(y_train_split)

    # Get the x_val_split
    x_val_split = x_train[val_index]
    x_val_split = tf.convert_to_tensor(x_val_split)

    # Get the y_train_split
    y_val_split = y_train[val_index]
    y_val_split = tf.convert_to_tensor(y_val_split)

    # Create the appropriate model
    temp_model = model.create_model(lr, drop, hidden)

    # Define the Tensorboard callback and name
    # log_dir = "/log/" + str(run_label) + "-" + str(hidden) + "-lr-" + str(lr) + "-drop-" + str(drop) + "-batch-" \
    #           + str(batch_size) + "/" + str(fold_var)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early stoppage
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1',
                                                      mode='max',
                                                      verbose=0,
                                                      patience=50)

    # CREATE CALLBACKS
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    model_save_dir = os.path.join('saved_models',
                                  str(run_label) + "-" + str(hidden) + "-lr-" + str(lr) + "-drop-" + str(drop)
                                  + "-batch-" + str(batch_size),
                                  str(fold_var))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_dir, monitor='val_f1', verbose=0, save_best_only=True,
                                                    mode='max')

    # Train the model
    temp_model.fit(x=x_train_split, y=y_train_split,
                   validation_data=(x_val_split, y_val_split),
                   batch_size=batch_size,
                   epochs=1000,
                   verbose=0,
                   callbacks=[early_stopping, checkpoint])

    # Load in the best model
    temp_model = tf.keras.models.load_model(model_save_dir, custom_objects={'f1': f1, "precision": precision, 'recall': recall})
    temp_model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1, precision, recall])

    # Convert to tensors
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    stats = temp_model.evaluate(x=x_test, y=y_test, return_dict=True)

    VALIDATION_F1.append(stats['f1'])
    print(stats['f1'])
    VALIDATION_LOSS.append(stats['loss'])
    print(stats['loss'])

    fold_var += 1

print(VALIDATION_F1)
f = open(os.path.join('saved_models', str(run_label) + "-" + str(hidden) + "-lr-" + str(lr) + "-drop-" + str(drop)
                      + "-batch-" + str(batch_size), str(fold_var), 'results.txt'), "w+")
f.write('F1: ' + str(VALIDATION_F1))
f.write('loss: ' + str(VALIDATION_LOSS))

f.close()

VALIDATION_F1.sort()


