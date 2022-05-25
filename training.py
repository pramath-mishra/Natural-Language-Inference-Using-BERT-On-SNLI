import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import transformers
import tensorflow as tf
from data_gen import BertSemanticDataGenerator

# config
epochs = 2
batch_size = 32
max_length = 128

# labels
labels = ["contradiction", "entailment", "neutral"]

# loading train data
train = pd.read_csv(
    "SNLI_Corpus/snli_1.0_train.csv",
    low_memory=False,
    usecols=[
        "sentence1",
        "sentence2",
        "similarity"
    ]
)
train.dropna(axis=0, inplace=True)
train = train[train.similarity != "-"].sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"SNLI train data loaded...\n -shape: {train.shape}")

# loading validation data
val = pd.read_csv(
    "SNLI_Corpus/snli_1.0_dev.csv",
    low_memory=False,
    usecols=[
        "sentence1",
        "sentence2",
        "similarity"
    ]
)
val.dropna(axis=0, inplace=True)
val = val[val.similarity != "-"].sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"SNLI validation data loaded...\n -shape: {val.shape}")

# loading test data
test = pd.read_csv(
    "SNLI_Corpus/snli_1.0_test.csv",
    low_memory=False,
    usecols=[
        "sentence1",
        "sentence2",
        "similarity"
    ]
)
test.dropna(axis=0, inplace=True)
test = test[test.similarity != "-"].sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"SNLI test data loaded...\n -shape: {test.shape}")

# sample data
print(f"premise : {train.loc[1, 'sentence1']}")
print(f"hypothesis : {train.loc[1, 'sentence2']}")
print(f"label : {train.loc[1, 'similarity']}")

# one hot encoding labels
train["label"] = train["similarity"].apply(lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2)
y_train = tf.keras.utils.to_categorical(train.label, num_classes=3)

val["label"] = val["similarity"].apply(lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2)
y_val = tf.keras.utils.to_categorical(val.label, num_classes=3)

test["label"] = test["similarity"].apply(lambda x: 0 if x == "contradiction" else 1 if x == "entailment" else 2)
y_test = tf.keras.utils.to_categorical(test.label, num_classes=3)

# model under distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_masks")
    token_type_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # fine-tune BERT model
    bert_model.trainable = True
    bert_output = bert_model.bert(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

    # extracting embedding & passing to Bi-directional GRU
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True))(sequence_output)

    # hybrid pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_gru)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_gru)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])


print(f"Strategy: {strategy}")
print(model.summary())

# train and validation data generation
train_data = BertSemanticDataGenerator(
    train[["sentence1", "sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True
)
valid_data = BertSemanticDataGenerator(
    val[["sentence1", "sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)
print("train & validation data generated...")

# model training
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)

# model saving
model.save("./NLI_model", save_format='tf')
print("model saved...")

# model evaluation
test_data = BertSemanticDataGenerator(
    test[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
result = model.evaluate(test_data, verbose=1)
print(f"Accuracy: {result[1]}, Loss: {result[0]}")
