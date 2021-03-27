import math
import tensorflow as tf
from DataConnector import DataConnector
from EvaluationFunctions import f1, precision, recall
import os
import sqlite3
import numpy as np
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# Get unrated documents
db_path = 'reddit_50_both_inferred.db'
database = DataConnector(db_path)
documents = database.get_unrated_attitude_documents()

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()



# Load in the model
model = tf.keras.models.load_model(os.path.join("saved_models", "final-128-lr-0.001-drop-0.3-batch-50"),
                                   custom_objects={'f1': f1, 'precision': precision, 'recall': recall})
model.compile(run_eagerly=True, loss='binary_crossentropy', metrics=[f1, precision, recall])

# Update database with new column if needed
# TODO: check for duplicate column error
# query = 'ALTER TABLE comments ADD COLUMN inferred_attitude INTEGER'
# query = 'ALTER TABLE comments ADD COLUMN inferred_attitude_weight REAL'
# cursor.execute(query)
# query = 'ALTER TABLE comments ADD COLUMN inferred_persuasion INTEGER'
# cursor.execute(query)

print("Length: " + str(len(documents)))
batch_size = 5000
for i in range(math.ceil(len(documents)/batch_size)):

    # Time it
    start = time.time()

    # Get index slices
    first_index = i * batch_size
    last_index = min((i + 1) * batch_size, len(documents))

    # Slice the index
    document_batch = documents[first_index:last_index]

    # Get the activations from Roberta
    x = database.extract_activations_inference(document_batch, 5)

    model_input = np.array(x)

    # Get the inferred things
    inferred_classes = model.predict(model_input)

    for idx_, documentx in enumerate(document_batch):

        predicted_result = np.argmax(inferred_classes[idx_])
        query = "UPDATE comments SET inferred_attitude = {} WHERE ROWID == {}".format(int(predicted_result), int(documentx[0]))
        # print(query)
        # query = "UPDATE comments SET inferred_attitude = 0 WHERE ROWID == 5"
        cursor.execute(query)

        query = "UPDATE comments SET inferred_attitude_weight = {} WHERE ROWID == {}".format(
            inferred_classes[idx_][predicted_result], documentx[0])
        cursor.execute(query)

    conn.commit()

    # Time it
    end = time.time()

    # Verbose
    print("Inference performed for {}/{} documents in {} seconds.".format(last_index, len(documents), end - start))

cursor.close()
conn.close()


