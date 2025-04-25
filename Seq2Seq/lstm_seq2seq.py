"""
Title: Character-level recurrent sequence-to-sequence model
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2017/09/29
Last modified: 2023/11/22
Description: Character-level recurrent sequence-to-sequence model.
Accelerator: GPU
"""

"""
## Introduction

This example demonstrates how to implement a basic character-level
recurrent sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

**Summary of the algorithm**

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
"""

"""
## Setup
"""

import numpy as np
import keras
import os
from pathlib import Path
import requests
import zipfile

"""
## Download the data - Fixed version to handle HTTP 406 error
"""

# Create directory for data if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Define file paths
zip_path = data_dir / 'fra-eng.zip'
extract_dir = data_dir
data_file = extract_dir / 'fra.txt'

# Only download if the file doesn't exist
if not zip_path.exists():
    print("Downloading fra-eng.zip dataset...")
    url = "http://www.manythings.org/anki/fra-eng.zip"
    
    # Use requests with a user agent header
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    else:
        raise Exception(f"Failed to download file: Status code {response.status_code}")

# Extract the zip file if the data directory doesn't contain fra.txt
if not data_file.exists():
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete!")

# Set data path for use in the rest of the script
data_path = str(data_file)

"""
## Configuration
"""

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

"""
## Prepare the data
"""

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split("\t")
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = "\t" + target_text + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype="float32",
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype="float32",
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
    decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0

"""
## Build the model
"""

# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder = keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

"""
## Train the model
"""

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
model.save("s2s_model.keras")

"""
## Run inference (sampling)

1. encode input and retrieve initial decoder state
2. run one step of decoder with this initial state
and a "start of sequence" token as target.
Output will be the next target token.
3. Repeat with the current target token and current states
"""

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s_model.keras")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


"""
You can now generate decoded sentences as such:
"""

for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

"""
## Code Analysis by Section

* Lines 1-44: Introduction and description of the sequence-to-sequence algorithm
  - Provides an overview of character-level sequence-to-sequence translation
  - Explains the encoder-decoder LSTM architecture
  - Describes the training process ("teacher forcing") and inference process
  
* Lines 50-55: Importing necessary libraries
  - Imports numpy for numerical operations and array handling
  - Imports keras for deep learning model creation
  - Imports os and pathlib for file and directory operations
  - Imports requests and zipfile for downloading and extracting data
  
* Lines 57-97: Dataset download and extraction
  - Creates directory structure for data storage
  - Defines paths for the zip file and extracted data
  - Downloads the English-French dataset with proper headers
  - Extracts the zip file if needed
  - Sets the data path for use in the rest of the script
  
* Lines 100-108: Configuration settings
  - Sets batch size for training (64)
  - Sets number of training epochs (100)
  - Defines latent dimension size for LSTM (256)
  - Limits number of samples to use (10,000)
  
* Lines 109-167: Data preparation and preprocessing
  - Lines 105-126: Read data and create character sets
    - Opens and reads the dataset file
    - Splits lines and extracts input-target text pairs
    - Adds start/end sequence markers to target texts
    - Builds sets of unique characters for input and target languages
  
  - Lines 128-138: Creates vocabulary mappings and calculates statistics
    - Sorts character sets and gets vocabulary sizes
    - Creates dictionaries mapping characters to indices
    - Calculates maximum sequence lengths
    - Prints dataset statistics
  
  - Lines 140-168: Creates input and target data tensors
    - Creates one-hot encoded representations of input texts
    - Creates one-hot encoded representations of decoder inputs
    - Creates one-hot encoded representations of decoder targets (shifted by one timestep)
    - Pads sequences as needed
  
* Lines 170-199: Model building (encoder-decoder architecture)
  - Lines 172-179: Creates encoder part
    - Defines input layer for encoder
    - Creates LSTM layer that returns states
    - Extracts the final states from encoder output
  
  - Lines 181-199: Creates decoder part
    - Defines input layer for decoder
    - Creates LSTM layer with encoder states as initial state
    - Adds dense layer with softmax activation for character prediction
    - Combines encoder and decoder into a single training model
  
* Lines 201-214: Model training and saving
  - Compiles model with RMSprop optimizer and categorical crossentropy loss
  - Trains the model with the prepared data
  - Uses 20% of data for validation
  - Saves the trained model to disk
  
* Lines 216-273: Inference model creation
  - Lines 220-227: Loads the trained model
  - Lines 227-233: Creates encoder inference model
    - Uses the trained encoder from the main model
    - Sets up encoder model that outputs state vectors
  
  - Lines 235-254: Creates decoder inference model
    - Uses the trained decoder from the main model
    - Creates new input layers for states
    - Sets up decoder model that takes states and a character as input
    - Outputs character predictions and updated states
  
  - Lines 256-273: Creates helper functions for inference
    - Creates reverse index mappings from indices to characters
    - Defines decode_sequence function that implements the inference loop
      - Encodes input sequence to get initial states
      - Generates target sequence one character at a time
      - Uses beam search with argmax to select next character
      - Updates states and continues until end sequence or length limit
  
* Lines 275-285: Testing the translation system
  - Loops through 20 samples from the training set
  - Translates each input sentence using the trained model
  - Prints the original English and the translated French text
"""
