import tensorflow as tf
import numpy as np
import os
import time

from pathlib import Path 
file_path = Path.home() / ".keras" / "datasets" / "alice.txt"

# Load "Alice in Wonderland" text dataset
if file_path.exists():
    print("file is present")
else:
    print("file is not present")
    file_path = tf.keras.utils.get_file('alice.txt', 'https://www.gutenberg.org/files/11/11-0.txt')
text = open(file_path, 'rb').read().decode(encoding='utf-8')

# Preprocess the text
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])


# print(len(vocab))
# print(vocab)

# print(char2idx )
# print(idx2char)
print("length of file is " , len(text_as_int))

# Prepare training sequences
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
# print(examples_per_epoch)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# print(sequences.as_numpy_iterator().next() )


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
# print(dataset.as_numpy_iterator().next() )
print("***" , dataset )

# print(split_input_target("Amit"))    
# print(split_input_target("A"))    
# print(split_input_target("Amiscefhvbiewvbrevbiuigvhbfejnvbfjbvfj"))    

# Create training batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print("***___" , dataset )

# Build the RNN model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
