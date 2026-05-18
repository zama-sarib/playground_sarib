import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================================================
# Model Folder
# =========================================================

# IMPORTANT:
# Use the exact training date folder
date_folder = "2026-05-18"

base_dir = r"C:\Users\zamas\Desktop\Sarib Workspace\playground_sarib\artefact"

model_dir = os.path.join(
    base_dir,
    "saved_models",
    date_folder
)

# =========================================================
# Load Saved Artifacts
# =========================================================

model = load_model(
    os.path.join(model_dir, "seq2seq_model.h5")
)

print("Model loaded successfully!")

with open(
    os.path.join(model_dir, "input_tokenizer.pkl"),
    "rb"
) as f:

    input_tokenizer = pickle.load(f)

print("Input tokenizer loaded successfully!")

with open(
    os.path.join(model_dir, "target_tokenizer.pkl"),
    "rb"
) as f:

    target_tokenizer = pickle.load(f)

print("Target tokenizer loaded successfully!")

with open(
    os.path.join(model_dir, "config.pkl"),
    "rb"
) as f:

    config = pickle.load(f)

print("Config loaded successfully!")

# =========================================================
# Config Values
# =========================================================

max_encoder_len = config["max_encoder_len"]

max_decoder_len = config["max_decoder_len"]

latent_dim = config["latent_dim"]

# =========================================================
# Reverse Dictionaries
# =========================================================

reverse_target_word_index = {
    i: word
    for word, i in target_tokenizer.word_index.items()
}

target_word_index = target_tokenizer.word_index

# =========================================================
# Print Layers (Optional Debugging)
# =========================================================

print("\nModel Layers:\n")

for i, layer in enumerate(model.layers):

    print(i, layer.name)

# =========================================================
# Encoder Inference Model
# =========================================================

encoder_inputs = model.input[0]

encoder_embedding_layer = model.get_layer(
    "encoder_embedding"
)

encoder_lstm = model.get_layer(
    "encoder_lstm"
)

encoder_embedding = encoder_embedding_layer(
    encoder_inputs
)

encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(
    encoder_inputs,
    encoder_states
)

print("\nEncoder model built successfully!")

# =========================================================
# Decoder Inference Model
# =========================================================

decoder_inputs = model.input[1]

decoder_embedding_layer = model.get_layer(
    "decoder_embedding"
)

decoder_lstm = model.get_layer(
    "decoder_lstm"
)

decoder_dense = model.get_layer(
    "decoder_dense"
)

# Decoder states inputs
decoder_state_input_h = Input(
    shape=(latent_dim,)
)

decoder_state_input_c = Input(
    shape=(latent_dim,)
)

decoder_states_inputs = [
    decoder_state_input_h,
    decoder_state_input_c
]

# Decoder embedding
decoder_embedding = decoder_embedding_layer(
    decoder_inputs
)

# Decoder LSTM
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embedding,
    initial_state=decoder_states_inputs
)

decoder_states = [state_h_dec, state_c_dec]

# Dense layer
decoder_outputs = decoder_dense(
    decoder_outputs
)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

print("Decoder model built successfully!")

# =========================================================
# Decode Function
# =========================================================

def decode_sequence(input_sentence):

    # -----------------------------------------------------
    # Tokenize Input Sentence
    # -----------------------------------------------------

    input_seq = input_tokenizer.texts_to_sequences(
        [input_sentence]
    )

    input_seq = pad_sequences(
        input_seq,
        maxlen=max_encoder_len,
        padding='post'
    )

    # -----------------------------------------------------
    # Encode Input Sentence
    # -----------------------------------------------------

    states_value = encoder_model.predict(
        input_seq,
        verbose=0
    )

    # -----------------------------------------------------
    # Initialize Decoder Input
    # -----------------------------------------------------

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = target_word_index["<start>"]

    # -----------------------------------------------------
    # Decoding Loop
    # -----------------------------------------------------

    decoded_sentence = ""

    stop_condition = False

    while not stop_condition:

        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value,
            verbose=0
        )

        # Get predicted token index
        sampled_token_index = np.argmax(
            output_tokens[0, -1, :]
        )

        # Convert token index -> word
        sampled_word = reverse_target_word_index.get(
            sampled_token_index,
            ""
        )

        # -------------------------------------------------
        # Stop Conditions
        # -------------------------------------------------

        if (
            sampled_word == "<end>"
            or len(decoded_sentence.split()) > max_decoder_len
        ):

            stop_condition = True

        else:

            decoded_sentence += sampled_word + " "

        # -------------------------------------------------
        # Update Decoder Input
        # -------------------------------------------------

        target_seq = np.zeros((1, 1))

        target_seq[0, 0] = sampled_token_index

        # -------------------------------------------------
        # Update States
        # -------------------------------------------------

        states_value = [h, c]

    return decoded_sentence.strip()

# =========================================================
# Interactive Translation
# =========================================================

print("\nEnglish to Hindi Translation Ready!")
print("Type 'exit' to quit.\n")

while True:

    sentence = input("Enter English Sentence: ")

    if sentence.lower() == "exit":

        print("\nExiting...")
        break

    translation = decode_sequence(sentence)

    print("\nHindi Translation:")
    print(translation)
    print()