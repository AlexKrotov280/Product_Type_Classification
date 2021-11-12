import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from pymongo import MongoClient
import pandas as pd
import numpy as np
import pickle

# # It was added in Pycharm only to work with memory correctly
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

vocab_size = 5000
embedding_dim = 64
max_length = 7
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.9

MODEL_PATH = r"D:\FOOTY\ProductTypeClassification\NLP"
TOKENIZER_PATH = r"D:\FOOTY\ProductTypeClassification\NLP\tokenizer.pickle"

myclient = MongoClient("XXX", username='USER_NAME', password='PASSWORD')
db = myclient["footy_live"]
collection = db["Product"]
cursor = collection.aggregate([
    {"$project": {"Description": "$Name", "Product_type": "$Attributes.ProductType", "_id": 0}}])
df = pd.DataFrame(list(cursor))
df.dropna(inplace=True)
df["Description"] = df["Description"].astype(str)
df["Product_type"] = df["Product_type"].astype(str)

unique_name_list = ["Trainers", "T-Shirt", "Replica Shirt", "Boots", "Hoodie",
                    "Jacket", "Shorts", "Replica Shorts", "Training Pants", "Bag",
                    "Training Top", "Polo Shirt", "Hat", "Tracksuit", "Replica Socks",
                    "Sweatshirt", "Tank Top", "Football", "Baselayer", "Leggings", "Socks",
                    "Goalkeeper Gloves", "Replica Mini Kit", "Shin Guards", "Belt", "Gloves",
                    "Keyring", "Mug", "Scarf", "Sliders","Sunglasses", "Water Bottle"]

df = df[df['Product_type'].isin(unique_name_list)]

df["Product_type"].replace({"Goalkeeper Gloves": "GoalkeeperGloves",
                            "Polo Shirt": "PoloShirt",
                            "Replica Mini Kit":"ReplicaMiniKit",
                            "Replica Shirt":"ReplicaShirt",
                            "Replica Shorts":"ReplicaShorts",
                            "Shin Guards":"ShinGuards",
                            "T-Shirt":"TShirt",
                            "Tank Top":"TankTop",
                            "Training Pants":"TrainingPants",
                            "Water Bottle":"WaterBottle",
                            "Replica Socks":"ReplicaSocks",
                            "Training Top":"TrainingTop",
                            "Water Bottle":"WaterBottle"}, inplace=True)

df = df.sample(frac=1)

description_list = df['Description'].values.tolist()
product_type_list = df['Product_type'].values.tolist()
print(len(description_list), set(product_type_list))

train_size = int(len(description_list) * training_portion)

train_description = description_list[0: train_size]
train_labels = product_type_list[0: train_size]

validation_description = description_list[train_size:]
validation_labels = product_type_list[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(train_description)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_description)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_description)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(product_type_list)
print(list(label_tokenizer.word_index.keys()))

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(33, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 7
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)


model.save(MODEL_PATH)

with open(TOKENIZER_PATH, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)