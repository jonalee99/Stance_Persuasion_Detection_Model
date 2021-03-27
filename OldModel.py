from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from transformers import RobertaConfig, RobertaTokenizer, TFRobertaModel, pipeline
from collections import Counter
from EvaluationFunctions import f1, precision, recall, write_metrics

import tensorflow as tf
import sqlite3
import os
import sys
import numpy as np
import time
import copy
import tables
import matplotlib.pyplot as plt
import datetime


class Model(object):
    def __init__(self):
        self.num_topics = 50
        self.top_subs = 500
        self.top_authors = 200
        self.authorship = True
        self.LDA_Topics = True
        self.use_subreddits = True
        self.DOI = "attitude"
        self.l2_regularization = False
        self.model = None
        self.input1 = None
        self.input2 = None
        self.ff1 = None
        self.out = None
        self.epochs = 1000
        self.batch_size = 50
        self.num_classes = 2
        self.cp_callback = None
        self.early_stopping = None
        self.validation_split = 0.2
        self.path_to_drive = "/content/drive/My Drive/Reddit_Marijuana_Legalization_Corpus/"
        self.verbose = 1
        self.train_flag = 1
        self.test_flag = 0
        self.persuasion_reduced_class_mapping = {0: 0,
                                                 1: 1,
                                                 2: 1,
                                                 3: 1}

        self.attitude_reduced_class_mapping = {1: 0,
                                               2: 0,
                                               0: 0,
                                               3: 0,
                                               4: 1,
                                               5: 1}  # maps the old label to the index for the new label

    '''
    Function to extract the model checkpoint path based 
    on the user-defined configurables
    @return the path to a directory where the model checkpoints exist
    '''

    def get_checkpoint_path(self):
        configurable_paths = []
        if self.LDA_Topics:
            configurable_paths.append("LDA")
        if self.authorship:
            configurable_paths.append("authorship")
        if self.use_subreddits:
            configurable_paths.append("subreddit")
        checkpoint_path = "{}/Trained_Models/Main_NN/{}{}{}".format(self.path_to_drive,
                                                                    "-".join(configurable_paths),
                                                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                                    "training_1/checkpoint")
        return os.path.dirname(checkpoint_path)

    def plot_histogram(self):
        conn = sqlite3.connect(self.path_to_drive + "Full_Data_Cleaned_8.5.2020/reddit_50.db")
        cursor = conn.cursor()
        persuasion = []
        attitude = []
        sql_attitude = "SELECT attitude FROM comments WHERE attitude IS NOT NULL"
        sql_persuasion = "SELECT persuasion FROM comments WHERE persuasion IS NOT NULL"
        cursor.execute(sql_persuasion)
        for val in cursor.fetchall():
            arr = val[0].split(",")
            for at in arr:
                attitude.append(at)

        at_hist = plt.hist(attitude, bins='auto')
        plt.title("Persuasion Ratings")
        plt.show()

    '''
    Function to extract the documents based on the configurable input
    @param get_rated a boolean value indicating whether to get rated, or unrated data
    @return a list of document tuples, where each tuple contains 
    the relevant configurable input (e.g LDA, author, subreddit, etc) info
    for that particular document
    '''

    def extract_documents(self, get_rated):
        try:
            conn = sqlite3.connect(self.path_to_drive + "/Full_Data_Cleaned_8.5.2020/reddit_50.db")
            cursor = conn.cursor()

            # Baseline fields from the database we'll be extracting
            if get_rated:
                fields = ["ROWID", "original_comm", self.DOI, "training"]
            else:
                fields = ["ROWID", "original_comm"]

            if self.LDA_Topics:
                # Add self.num_topics topic columns to the fields
                for topic in range(self.num_topics):
                    fields.append("topic_{}".format(topic))

                # If we are adding authorship
            if self.authorship:
                # Set the authorship one hot vector to size = number of authors + 1
                # Add an extra cell for authors not in our top author set/deleted authors
                self.authorship_one_hot = [0 for i in range(self.top_authors + 1)]

                # Extract the most prolific authors, based on self.top_authors
                cursor.execute(
                    "SELECT author, COUNT(*) FROM comments GROUP BY author ORDER BY COUNT(*) DESC LIMIT {}".format(
                        self.top_authors + 1))

                # Initialize a set for uniqueness, and wrap it
                # in a list for ordered indexing
                self.top_authors_list = list(set([item[0] for item in cursor.fetchall()]))

                fields.append("author")

            # If we are adding subreddit info
            if self.use_subreddits:
                # Set the subreddit one hot vector to size = number of subreddits + 1
                # Add an extra cell for subreddits not in our top subreddit set
                self.subreddits_one_hot = [0 for i in range(self.top_subs + 1)]
                cursor.execute(
                    "SELECT subreddit, COUNT(*) FROM comments GROUP BY subreddit ORDER BY COUNT(*) DESC LIMIT {}".format(
                        self.top_subs))

                # Initialize a set for uniqueness, and wrap it
                # in a list for ordered indexing
                self.top_subreddits_list = list(set([item[0] for item in cursor.fetchall()]))

                fields.append("subreddit")

            self.fields = fields

            # Extract all the documents with the fields added from above
            final_query = ""
            # If we are getting rated data
            if get_rated:
                final_query = "SELECT {} FROM comments WHERE {} IS NOT NULL".format(",".join(self.fields), self.DOI)
            else:
                final_query = "SELECT {} FROM comments WHERE {} IS NULL".format(",".join(self.fields), self.DOI)

            cursor.execute(final_query)
            # Return all the document tuples, the configurable fields,
            # the authorship and subreddit lists
            return cursor.fetchall()

        except sqlite3.Error as error:
            print("Failed to read data from table", error)

    '''
    Function to extract the input vectors from a given batch
    @param document_batch, a list of tuples with each input type
    @param train_or_test, a flag indicating whether we are looking for training or testing data
    @return a numpy array of arrays, where each inner array is of size input2_size
    '''

    def extract_input_vectors(self, document_batch, train_flag):

        # define the Roberta objects
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta = TFRobertaModel.from_pretrained('roberta-base')

        # # Open activations file
        # h5 = tables.open_file(self.path_to_drive +
        #                       "Full_Data_Cleaned_8.5.2020/Roberta_Set/Roberta_set.h5", 'r')
        # carr = h5.root.carray
        # print(len(carr))

        activations = []
        additional_input = []
        output = []
        # For each document in the batch
        for tup in document_batch:
            # Each document will have its own row for input. We add to it as
            # we loop through the columns
            additional_input_component = []
            LDA_Vec = []

            # If the cell corresponds to the training/testing set, or we are not getting
            # rated data
            if train_flag is None or tup[3] == train_flag:

                for (index, column) in enumerate(tup):
                    column_name = self.fields[index]
                    # if column_name == "ROWID":
                    if column_name == "original_comm":
                        # Get the activations for the appropriate row
                        # activations.append(carr[column - 1,:]) # Should be a vector of size
                        encoded_input = tokenizer(tup[1], return_tensors="tf", truncation=True, padding=True,
                                                  max_length=512)
                        roberta_output = roberta(encoded_input)
                        activations.append(tf.math.reduce_mean(roberta_output[0][0],
                                                               axis=0))  # pooler output; shape (batch_size, hidden_size)

                    if column_name == "author":
                        # if author in top authors, create vector and append to additional
                        # input component
                        new_author_vec = copy.deepcopy(self.authorship_one_hot)
                        if column in self.top_subreddits_list:
                            index = self.top_authors_list.index(column)
                            new_author_vec[index] = 1
                        else:
                            new_author_vec[0] = 1

                        # Add the author vector to the additional input component
                        for num in new_author_vec:
                            additional_input_component.append(num)

                    if column_name == "subreddit":
                        # if subreddit in top subreddits, create vector and append
                        # to additional input component
                        new_subreddit_vec = copy.deepcopy(self.subreddits_one_hot)
                        if column in self.top_subreddits_list:
                            index = self.top_subreddits_list.index(column)
                            new_subreddit_vec[index] = 1
                        else:
                            new_subreddit_vec[len(new_subreddit_vec) - 1] = 1

                        # Add the subtreddit vector to the additional input component
                        for num in new_subreddit_vec:
                            additional_input_component.append(num)

                    # If we are on a topic column
                    if "topic" in column_name:
                        # And the contribution value is not null, append the value
                        # otherwise, just append zero
                        if column:
                            LDA_Vec.append(float(column))
                        else:
                            LDA_Vec.append(0.0)

                    # If we are on a label column, append that to the output
                    if column_name == self.DOI:
                        output_array = [0] * self.num_classes
                        # If we just have one rating, create one-hot vector
                        if isinstance(column, int):
                            output_array[self.attitude_reduced_class_mapping[column] if self.DOI == "attitude" else
                            self.persuasion_reduced_class_mapping[column]] = 1
                            output.append(output_array)

                        # Otherwise, create a multi-hot vector proportional
                        # to the ratings
                        else:

                            reduced_ratings = [
                                self.attitude_reduced_class_mapping[int(i)] if self.DOI == "attitude" else
                                self.persuasion_reduced_class_mapping[int(i)]
                                for i in column.split(",")]
                            for r in reduced_ratings:
                                output_array[int(r)] = 1
                            output.append(output_array)

            # Add all the topic values to our additional input column
            for topic in LDA_Vec:
                additional_input_component.append(topic)
            additional_input.append(additional_input_component)

        return activations, additional_input, output

    '''
    Function to initialize the neural network. 
    '''

    def initialize_model(self, lr, dropout, hidden):
        self.model = keras.Sequential(
            [
                layers.Dense(hidden, activation="relu", name="layer1"),
                layers.Dropout(dropout),
                layers.Dense(self.num_classes, activation="softmax", name="layer2"),
            ]
        )
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                           run_eagerly=True,
                           metrics=["mae", "acc", precision, recall, f1],
                           loss='binary_crossentropy')

    '''
    Function to load an existing model
    '''

    def load_model(self, path_to_model):
        self.model = tf.keras.models.load_model(path_to_model, custom_objects={"f1": f1,
                                                                               "precision": precision,
                                                                               "recall": recall}, compile=True,
                                                options=None)

    '''
    Function to infer dimension of interest for documents determined by 
    extract_documents()
    '''

    def infer_DOI(self):

        # Get unrated documents
        documents = self.extract_documents(False)

        conn = sqlite3.connect(self.path_to_drive + "/Full_Data_Cleaned_8.5.2020/reddit_50.db")
        cursor = conn.cursor()

        # Update database with new column if needed
        # TODO: check for duplicate column error
        # query = 'ALTER TABLE comments ADD COLUMN inferred_attitude INTEGER'
        # cursor.execute(query)
        # query = 'ALTER TABLE comments ADD COLUMN inferred_persuasion INTEGER'
        # cursor.execute(query)

        document_batch = []
        print("Length: " + str(len(documents)))
        for id_, document in enumerate(documents):
            document_batch.append(document)

            if len(document_batch) == self.batch_size:

                activations, \
                additional_input, _ = self.extract_input_vectors(document_batch, None)
                model_input = np.array(activations)

                # If we got any additional training input
                # append it to our input vector
                if len(additional_input) > 0:
                    np.append(model_input, additional_input)

                inferred_classes = self.model.predict(model_input)

                for idx_, document in enumerate(document_batch):
                    predicted_result = np.argmax(inferred_classes[idx_])
                    query = "UPDATE comments SET inferred_{} = {} WHERE ROWID == {}".format(self.DOI, predicted_result,
                                                                                            document[0])
                    cursor.execute(query)
                    conn.commit()

                print("Inference performed for {}/{} documents.".format(id_, len(documents)))
                document_batch = []
        # timer
        print("Finishing time:" + time.strftime('%l:%M%p, %m/%d/%Y'))

    '''
    Function to perform the main training and evaluation loop
    over each batch of documents
    '''

    def train(self, epoch):

        # Get rated documents
        documents = self.extract_documents(True)

        # Create a list of documents
        document_batch = []
        print("Length: " + str(len(documents)))
        for document in documents:
            document_batch.append(document)

        # Extracting training and testing input
        extraction = self.extract_input_vectors(document_batch, self.train_flag)
        training_activations = extraction[0]
        additional_training_input = extraction[1]
        training_output = extraction[2]

        # Convert to np arrays
        model_training_input = np.squeeze(np.array(training_activations))
        training_output = np.array(training_output)

        # If we got any additional training input
        # append it to our input vector
        if len(additional_training_input) > 0:
            np.append(model_training_input, additional_training_input)
        print("Input size: " + str(np.shape(model_training_input)))
        print("Input y size: " + str(np.shape(training_output)))

        # Callbacks
        # Tensorboard
        log_dir = "/content/drive/My Drive/Reddit_Marijuana_Legalization_Corpus/logs/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Early stoppage
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          verbose=0,
                                                          patience=300)

        # Model saving
        model_directory = "/content/drive/My Drive/Reddit_Marijuana_Legalization_Corpus/saved_models/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory,
                                                         save_weights_only=False,
                                                         verbose=1)

        # Split the data
        x_train, x_valid, y_train, y_valid = train_test_split(model_training_input, training_output, test_size=0.2,
                                                              shuffle=True)

        # check the validation set balance
        temp = [np.where(r == 1)[0][0] for r in y_valid]
        print(Counter(temp))

        # Augment the data
        # Turn y_train from a one-hot into normal vector
        y_normal = [np.where(r == 1)[0][0] for r in y_train]

        # See the training class imbalance
        print(Counter(y_normal))

        # Oversample
        x_train_balanced, y_train_balanced = SMOTE().fit_resample(x_train, y_normal)

        # See the training class imbalance once more
        print(Counter(y_train_balanced))

        # # Convert labels into one hot
        # y_train_balanced_hot = np.zeros((y_train_balanced.size, y_train_balanced.max() + 1))
        # y_train_balanced_hot[np.arange(y_train_balanced.size), y_train_balanced] = 1

        # # Check its shape
        # print(y_train_balanced_hot.shape)

        # # Train on the balanced set using the validation split from before
        # print(x_train_balanced.shape)
        # print(y_train_balanced_hot)

        self.model.fit(x=x_train,
                       y=y_train,
                       validation_data=(x_valid, y_valid),
                       batch_size=self.batch_size,
                       epochs=epoch,
                       verbose=1,
                       callbacks=[tensorboard_callback, cp_callback])

        # Timer
        print("Finishing time:" + time.strftime('%l:%M%p, %m/%d/%Y'))

    '''
    Function to evaluate the model
    '''

    def evaluate_model(self, model_directory=None, flag=0):
        # Load the model in
        if model_directory is not None:
            self.model = tf.keras.models.load_model(model_directory,
                                                    custom_objects={'precision': precision, 'recall': recall, 'f1': f1,
                                                                    'write_metrics': write_metrics})
            self.model.compile(run_eagerly=True,
                               loss='binary_crossentropy',
                               metrics=["mae", "acc", precision, recall, f1])

        # Extract the documents
        documents = self.extract_documents(True)
        eval_input, additional_eval_input, y = self.extract_input_vectors(documents, flag)
        y = np.array(y)
        x = np.array(eval_input)
        if len(additional_eval_input):
            np.append(x, additional_eval_input)
        print("X: " + str(np.shape(x)) + " Y: " + str(np.shape(y)))

        # Check class imbalance
        y_count = [np.where(r == 1)[0][0] for r in y]
        counter = Counter(y_count)
        print(counter)

        # If it is the training set, get a random subsample
        if flag == self.train_flag:
            _, x, _, y = train_test_split(x, y, test_size=0.2, shuffle=True)
            print("Shuffled sizes - X: " + str(np.shape(x)) + " Y: " + str(np.shape(y)))

        self.model.evaluate(x=x, y=y)

    def grid_search(self):
        # Grid search

        lists = [[0.01, 0.5, 128, 100],
                 [0.001, 0.3, 128, 50],
                 [0.0001, 0.4, 512, 500],
                 [0.001, 0.3, 256, 50],
                 [0.0001, 0.4, 256, 500],
                 [0.01, 0.5, 256, 100]]

        # Test different learning rates
        lr_list = [0.01, 0.001, 0.0001]

        # Test different dropouts
        drop_list = [0.3, 0.4, 0.5]

        # Test different batch sizes
        batch_size_list = [50, 100]

        # Test different hidden layer sizes
        hidden_list = [128, 512]

        # Get rated documents
        documents = self.extract_documents(True)

        # Create a list of documents
        document_batch = []
        print("Length: " + str(len(documents)))
        for document in documents:
            document_batch.append(document)

        # Extracting training and testing input
        extraction = self.extract_input_vectors(document_batch, self.train_flag)
        training_activations = extraction[0]
        additional_training_input = extraction[1]
        training_output = extraction[2]

        # Convert to np arrays
        model_training_input = np.squeeze(np.array(training_activations))
        training_output = np.array(training_output)

        # If we got any additional training input
        # append it to our input vector
        if len(additional_training_input) > 0:
            np.append(model_training_input, additional_training_input)
        print("Input size: " + str(np.shape(model_training_input)))
        print("Input y size: " + str(np.shape(training_output)))

        # Split the data
        x_train, x_valid, y_train, y_valid = train_test_split(model_training_input, training_output, test_size=0.2,
                                                              shuffle=True)

        # For all combinations
        for combination in lists:
            lr = combination[0]
            drop = combination[1]
            hidden = combination[2]
            batch_size = combination[3]

            # Create the appropriate model
            self.initialize_model(lr, drop, hidden)

            # Define the Tensorboard callback and name
            log_dir = "/content/drive/My Drive/Reddit_Marijuana_Legalization_Corpus/logs/grid6/" + self.DOI + "-hidden-" + str(
                hidden) + "-lr-" + str(lr) + "-drop-" + str(drop) + "-batch-" + str(batch_size)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            # Early stoppage
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_f1',
                                                              mode='max',
                                                              verbose=1,
                                                              patience=300,
                                                              restore_best_weights=True)

            # Train the model
            self.model.fit(x=x_train, y=y_train,
                           validation_data=(x_valid, y_valid),
                           batch_size=batch_size,
                           epochs=1000,
                           verbose=0,
                           callbacks=[tensorboard_callback, early_stopping])

            # Save the model
            self.model.save(
                "/content/drive/My Drive/Reddit_Marijuana_Legalization_Corpus/saved_models/grid6/" + self.DOI + "-hidden-" + str(
                    hidden) + "-lr-" + str(lr) + "-drop-" + str(drop) + "-batch-" + str(batch_size))