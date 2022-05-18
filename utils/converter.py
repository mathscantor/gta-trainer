import tensorflow as tf
import tensorflow_text as tf_text
import numpy as np
import time


class Converter:
    def __init__(self, max_characters):
        self.description = "Converts stuff"
        self.max_characters = max_characters


    def convert_domain_to_encoded_tensor(self,
            domain: str,
            sequence_length: int = 256,
            corpus_size: int = 10000) -> tf.Tensor:
        tokenizer = tf_text.UnicodeCharTokenizer()
        pad_token = tf.constant(0, dtype=tf.int32)

        tokens = tokenizer.tokenize([[domain]])
        tokens = tokens.merge_dims(1, 2)
        tokens = tokens[:, :sequence_length].to_tensor(default_value=pad_token)

        pad = sequence_length - tf.shape(tokens)[1]
        tokens = tf.pad(tokens, [[0, 0], [0, pad]], constant_values=pad_token)

        tokens = tf.reshape(tokens, [-1, sequence_length])
        tokens = tf.clip_by_value(tokens, clip_value_min=0, clip_value_max=corpus_size - 1)
        return tf.reshape(tf.cast(tokens, dtype="int32"), [1, self.max_characters])

    def convert_domains_to_encoded_tensor(self, domains) -> tf.Tensor:
        i = 1
        abs_num = 1
        domains_encoded_numpy = np.array([])
        temp_list = []
        for domain in domains:
            domain_encoded_numpy = self.convert_domain_to_encoded_tensor(domain).numpy()
            if i == 1:
                start_time = time.time()
                domains_encoded_numpy = domain_encoded_numpy
            else:
                domains_encoded_numpy = np.vstack([domains_encoded_numpy, domain_encoded_numpy])
            if i % 1000 == 0 or abs_num == len(domains):
                temp_list.append(domains_encoded_numpy)
                print("Batches Processed {}".format(len(temp_list)))
                print("--- %s seconds ---" % (time.time() - start_time))
                i = 0 # reset
            i += 1
            abs_num += 1

        domains_encoded_numpy = np.vstack(temp_list)
        return tf.convert_to_tensor(domains_encoded_numpy, dtype="int32")


    def convert_labels_to_tensor(self,labels):
        labels_binary = []
        labels
        for label in labels:
            if label == "legit":
                labels_binary.append(0)
            else:
                labels_binary.append(1)
        labels_binary = np.asarray(labels_binary)
        return tf.cast(tf.one_hot(tf.math.sign(labels_binary), 2), tf.int32)
