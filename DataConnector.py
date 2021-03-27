import sqlite3
import tensorflow as tf
import numpy as np
from transformers import RobertaTokenizer, TFRobertaModel
import math
import time
import re

class DataConnector:

    def __init__(self, database_path):
        self.database_path = database_path

    # Using the information defined above, this function returns a list of [COMMENT, RATINGS, TRAINING/TESTING]
    def get_documents(self):
        try:
            # Connect to the database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Vanilla model
            cursor.execute("SELECT original_comm, attitude, training FROM comments WHERE attitude IS NOT NULL")

            # Return all the document tuples, the configurable fields,
            # the authorship and subreddit lists
            return cursor.fetchall()

        except sqlite3.Error as error:
            print("Failed to read data from table", error)

    # Using the information defined above, this function returns a list of [COMMENT, RATINGS, TRAINING/TESTING]
    def get_documents_persuasion(self):
        try:
            # Connect to the database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Vanilla model
            cursor.execute("SELECT original_comm, persuasion, training FROM comments WHERE persuasion IS NOT NULL")

            # Return all the document tuples, the configurable fields,
            # the authorship and subreddit lists
            return cursor.fetchall()

        except sqlite3.Error as error:
            print("Failed to read data from table", error)

    def get_unrated_attitude_documents(self):
        try:
            # Connect to the database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Vanilla model
            cursor.execute("SELECT ROWID, original_comm FROM comments WHERE attitude IS NULL AND inferred_attitude IS NULL")

            # Return all the document tuples, the configurable fields,
            # the authorship and subreddit lists
            results = cursor.fetchall()

            cursor.close()
            conn.close()

            return results

        except sqlite3.Error as error:
            print("Failed to read data from table", error)

    def get_unrated_persuasion_documents(self):
        try:
            # Connect to the database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()

            # Vanilla model
            cursor.execute("SELECT ROWID, original_comm FROM comments WHERE persuasion IS NULL AND inferred_persuasion IS NULL")

            # Return all the document tuples, the configurable fields,
            # the authorship and subreddit lists
            results = cursor.fetchall()

            cursor.close()
            conn.close()

            return results

        except sqlite3.Error as error:
            print("Failed to read data from table", error)

    '''
    This extracts the activations of a document batch
    document_batch: (COMMENT, RATING, TRAINING)
    train_flag: int which represents the training flag
    returns: activations, one-hot labels
    '''
    def extract_activations(self, document_batch, train_flag):

        # Create the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta = TFRobertaModel.from_pretrained('roberta-base')

        # These lists hold the activations and stuff
        activations = []
        output = []

        # Tokenize the vectors
        document_batch = np.array(document_batch)

        # Open the text file filled with weed words
        txt_file = open("MarijuanaTerms", "r")

        # load in the regex string
        regex_string = ""
        for index, word in enumerate(txt_file.read().splitlines()):
            if index == 0:
                regex_string = word.lower()
            else:
                regex_string = regex_string + "|" + word.lower()

        txt_file.close()

        # Keep track of length
        len_array = []

        # For each document in the batch
        for index, document in enumerate(document_batch):

            # If the cell corresponds to the training/testing set
            if int(document[2]) == train_flag:

                # book-keeping
                comment = document[0]

                # max_length = 2000
                # if len(comment) > max_length:
                #     comment = self.regex_clipper(document[0], regex_string, max_length)



                rating = document[1]

                # Tokenize the comment
                encoded_input = tokenizer(comment, return_tensors="tf", truncation=True, padding=True, max_length=512)

                # encoded_input = tokenizer(comment)
                # len_array.append(len(encoded_input['input_ids']))

                # Get the roberta output
                roberta_output = roberta(encoded_input)

                # Append it to the activations list
                activations.append(tf.math.reduce_mean(roberta_output[0][0], axis=0))

                # Append it to the output vector
                output.append(rating)

        return activations, output

    def extract_activations_inference(self, documents, batch_size):

        # Create the tokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta = TFRobertaModel.from_pretrained('roberta-base')

        # Set the batch size
        batch_size = batch_size

        # Track output
        output_array = None

        # Time it
        # start = time.time()

        # Create the batches of documents
        for i in range(math.ceil(len(documents)/batch_size)):

            # Verbose
            if i % 100 == 0:
                print("Batch {}/{}".format(i + 1, math.ceil(len(documents)/batch_size)))

            # Get index slices
            first_index = i * batch_size
            last_index = min((i + 1) * batch_size, len(documents))

            # Slice the index
            document_batch = documents[first_index:last_index]

            # Isolate the comment
            comment_batch = np.array(document_batch)[:, 1]

            # Get the roberta output
            roberta_output = roberta(tokenizer(comment_batch.tolist(), return_tensors="tf", truncation=True, padding=True, max_length=512))

            # Append it to the activations list
            if output_array is None:
                output_array = tf.math.reduce_mean(roberta_output[0], axis=1)
            else:
                output_array = tf.concat([output_array, tf.math.reduce_mean(roberta_output[0], axis=1)], 0)

        # Time it
        # end = time.time()
        # print(end - start)

        return output_array

    # Adjust based on the key
    def one_hot_labels(self, labels, num_classes, attitude_mapping):

        output = []
        for rating in labels:

            # Create the one hot
            output_array = [0] * num_classes

            # If we just have one rating, create one-hot vector
            if isinstance(rating, int):

                # Put a 1 where it needs to be
                output_array[attitude_mapping[int(rating)]] = 1

                # Append it to the output vector
                output.append(output_array)

            # Otherwise, create a multi-hot vector proportional to the ratings
            else:

                # Split the ratings
                split_ratings = rating.split(",")

                # Put the 1 where it needs to be
                for r in split_ratings:
                    output_array[attitude_mapping[int(r)]] = 1

                # Append it to the output vector
                output.append(output_array)

        return output

    # this will return a 512 length clipped comment with the first instance of weed word at the center
    def regex_clipper(self, comment, regex_string, temp_len):

        # apply it to the input string
        a = re.search(regex_string, comment.lower())

        if a:

            temp_len = int(temp_len/2)

            # find the first instance of weed mention
            relevant_index = a.start()

            # If it's too close to the first word
            if relevant_index - temp_len < 0:
                relevant_index = temp_len

            # If it's too close to the last word
            if relevant_index + temp_len > len(comment):
                relevant_index = len(comment) - temp_len

            # Get the new comment
            new_comment = comment[max(0, relevant_index - temp_len): min(relevant_index + temp_len, len(comment))]

            # Adjust the indices to be on a word
            first_index = 0
            while new_comment[first_index] != ' ':
                first_index += 1
            last_index = len(new_comment) - 1
            while new_comment[last_index] != ' ':
                last_index -= 1

            return new_comment[first_index+1:last_index]

        else:
            return comment
