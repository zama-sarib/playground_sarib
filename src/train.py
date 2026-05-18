import pandas as pd
import os
import numpy as np
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datetime import datetime

# =========================================================
# Date Folder
# =========================================================

date_folder = datetime.now().strftime("%Y-%m-%d")

# =========================================================
# Load Dataset
# =========================================================

df = pd.read_csv(
    r"C:\Users\zamas\Desktop\Sarib Workspace\playground_sarib\Data\english_hindi_dataset.csv"
)

input_texts = df['english'].astype(str).tolist()

target_texts = [
    "<start> " + text + " <end>"
    for text in df['hindi'].astype(str).tolist()
]

# =========================================================
# Tokenization
# =========================================================

input_tokenizer = Tokenizer(filters='')

target_tokenizer = Tokenizer(filters='')

input_tokenizer.fit_on_texts(input_texts)

target_tokenizer.fit_on_texts(target_texts)

input_sequences = input_tokenizer.texts_to_sequences(
    input_texts
)

target_sequences = target_tokenizer.texts_to_sequences(
    target_texts
)

# =========================================================
# Vocabulary Sizes
# =========================================================

num_encoder_tokens = len(
    input_tokenizer.word_index
) + 1

num_decoder_tokens = len(
    target_tokenizer.word_index
) + 1

# =========================================================
# Padding
# =========================================================

max_encoder_len = max(
    len(seq) for seq in input_sequences
)

max_decoder_len = max(
    len(seq) for seq in target_sequences
)

encoder_input_data = pad_sequences(
    input_sequences,
    maxlen=max_encoder_len,
    padding='post'
)

decoder_input_data = pad_sequences(
    [seq[:-1] for seq in target_sequences],
    maxlen=max_decoder_len - 1,
    padding='post'
)

decoder_target_data = pad_sequences(
    [seq[1:] for seq in target_sequences],
    maxlen=max_decoder_len - 1,
    padding='post'
)

# =========================================================
# Better Memory Efficient Training
# Using Sparse Targets Instead of One Hot Encoding
# =========================================================

decoder_target_data = np.expand_dims(
    decoder_target_data,
    -1
)

# =========================================================
# Model Parameters
# =========================================================

embedding_dim = 128
latent_dim = 256

# =========================================================
# Encoder
# =========================================================

encoder_inputs = Input(
    shape=(None,),
    name="encoder_inputs"
)

encoder_embedding_layer = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=embedding_dim,
    name="encoder_embedding"
)

encoder_embedding = encoder_embedding_layer(
    encoder_inputs
)

encoder_lstm = LSTM(
    latent_dim,
    return_state=True,
    name="encoder_lstm"
)

encoder_outputs, state_h, state_c = encoder_lstm(
    encoder_embedding
)

encoder_states = [state_h, state_c]

# =========================================================
# Decoder
# =========================================================

decoder_inputs = Input(
    shape=(None,),
    name="decoder_inputs"
)

decoder_embedding_layer = Embedding(
    input_dim=num_decoder_tokens,
    output_dim=embedding_dim,
    name="decoder_embedding"
)

decoder_embedding = decoder_embedding_layer(
    decoder_inputs
)

decoder_lstm = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)

decoder_dense = Dense(
    num_decoder_tokens,
    activation='softmax',
    name="decoder_dense"
)

decoder_outputs = decoder_dense(
    decoder_outputs
)

# =========================================================
# Seq2Seq Model
# =========================================================

model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

# =========================================================
# Compile
# =========================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================================
# Summary
# =========================================================

model.summary()

# =========================================================
# Create Save Directory
# =========================================================

base_dir = (
    r"C:\Users\zamas\Desktop\Sarib Workspace"
    r"\playground_sarib\artefact"
)

save_dir = os.path.join(
    base_dir,
    "saved_models",
    date_folder
)

os.makedirs(save_dir, exist_ok=True)

# =========================================================
# Train
# =========================================================

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=32,
    epochs=50,
    validation_split=0.2
)

# =========================================================
# Save Model
# =========================================================

model_path = os.path.join(
    save_dir,
    "seq2seq_model.h5"
)

model.save(model_path)

# =========================================================
# Save Tokenizers
# =========================================================

with open(
    os.path.join(save_dir, "input_tokenizer.pkl"),
    "wb"
) as f:

    pickle.dump(input_tokenizer, f)

with open(
    os.path.join(save_dir, "target_tokenizer.pkl"),
    "wb"
) as f:

    pickle.dump(target_tokenizer, f)

# =========================================================
# Save Config
# =========================================================

config = {
    "max_encoder_len": max_encoder_len,
    "max_decoder_len": max_decoder_len,
    "latent_dim": latent_dim,
    "embedding_dim": embedding_dim
}

with open(
    os.path.join(save_dir, "config.pkl"),
    "wb"
) as f:

    pickle.dump(config, f)

# =========================================================
# Done
# =========================================================

print("\nTraining completed successfully!")
print(f"Artifacts saved to:\n{save_dir}")