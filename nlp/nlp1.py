"""
Title: Text classification using Decision Forests and pretrained embeddings
Usage: source venv/bin/activate
pip install -r requirements.txt
python nlp1.py

"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt

# Turn .csv files into pandas DataFrame's

PRETRAINED_EMBEDDINGS = False
TRAIN_FILE = "nlp1_train.csv"  # this is used for both training & test data


df = pd.read_csv(
    TRAIN_FILE
)
print(df.head())
print(f"Training dataset shape: {df.shape}")

"""
Shuffling and dropping unnecessary columns:
"""

df_shuffled = df.sample(frac=1, random_state=42)
# Dropping id, keyword and location columns as these columns consists of mostly nan values
# we will be using only text and target columns
df_shuffled.drop(["id"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())

"""
Printing information about the shuffled dataframe:
"""

print(df_shuffled.info())

"""
Total number of "disaster" and "non-disaster" tweets:
"""

print(
    "Total Number of disaster and non-disaster tweets: "
    f"{df_shuffled.target.value_counts()}"
)

"""
preview a few samples:
"""

for index, example in df_shuffled[:5].iterrows():
    print(f"Example #{index}")
    print(f"\tTarget : {example['target']}")
    print(f"\tText : {example['text']}")

"""
Splitting dataset into training and test sets:
"""
test_df = df_shuffled.sample(frac=0.3, random_state=42)
train_df = df_shuffled.drop(test_df.index)
test_records = len(test_df)
print(f"Using {len(train_df)} samples for training and {test_records} for validation")

"""
Total number of "disaster" and "non-disaster" tweets in the training data:
"""
print(train_df["target"].value_counts())

"""
Total number of "disaster" and "non-disaster" tweets in the test data:
"""

print(test_df["target"].value_counts())

"""
## Convert data to a `tf.data.Dataset`
"""


def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].to_numpy(), dataframe["target"].to_numpy())
    )
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)

"""
## Downloading pretrained embeddings

The Universal Sentence Encoder embeddings encode text into high-dimensional vectors that can be
used for text classification, semantic similarity, clustering and other natural language
tasks. They're trained on a variety of data sources and a variety of tasks. Their input is
variable-length English text and their output is a 512 dimensional vector.

To learn more about these pretrained embeddings, see
[Universal sentence encoder collections] (https://tfhub.dev/google/collections/universal-sentence-encoder/1)
[Universal Sentence Encoder - text embedding](https://tfhub.dev/google/universal-sentence-encoder/4).

"""

if PRETRAINED_EMBEDDINGS:
    sentence_encoder_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4"
    )
    inputs = layers.Input(shape=(), dtype=tf.string)
    outputs = sentence_encoder_layer(inputs)
    preprocessor = keras.Model(inputs=inputs, outputs=outputs)
else:
    preprocessor = None
model_1 = tfdf.keras.GradientBoostedTreesModel(preprocessing=preprocessor)

"""
Building model_2
"""

# model_2 = tfdf.keras.GradientBoostedTreesModel()

"""
## Train the models

We compile our model by passing the metrics `Accuracy`, `Recall`, `Precision` and
`AUC`. When it comes to the loss, TF-DF automatically detects the best loss for the task
(Classification or regression). It is printed in the model summary.

Also, because they're batch-training models rather than mini-batch gradient descent models,
TF-DF models do not need a validation dataset to monitor overfitting, or to stop
training early. Some algorithms do not use a validation dataset (e.g. Random Forest)
while some others do (e.g. Gradient Boosted Trees). If a validation dataset is
needed, it will be extracted automatically from the training dataset.
"""

# Compiling model_1
model_1.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
model_1.fit(train_ds)

# Compiling model_2
# model_2.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])
# Here we do not specify epochs as, TF-DF trains exactly one epoch of the dataset
# model_2.fit(train_ds)

"""
Prints training logs of model_1
"""

logs_1 = model_1.make_inspector().training_logs()
# print(logs_1)

"""
Prints training logs of model_2
"""

# logs_2 = model_2.make_inspector().training_logs()
# print(logs_2)

"""
The model.summary() method prints a variety of information about your decision tree model, including model type, task, input features, and feature importance.
"""

print("model_1 summary: ")
print(model_1.summary())
print()
# print("model_2 summary: ")
# print(model_2.summary())

"""
## Plotting training metrics
"""


def plot_curve(logs):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Loss")

    plt.show()


# plot_curve(logs_1)
# plot_curve(logs_2)

"""
## Evaluating on test data
"""

# results = model_1.evaluate(test_ds, return_dict=True, verbose=0)
print("model_1 Evaluation: \n")
#for name, value in results.items():
#    print(f"{name}: {value:.4f}")

# results = model_2.evaluate(test_ds, return_dict=True, verbose=0)
# print("model_2 Evaluation: \n")
#for name, value in results.items():
#    print(f"{name}: {value:.4f}")

"""
## Predicting on validation data
"""


test_df.reset_index(inplace=True, drop=True)
score = 0
for index, row in test_df.iterrows():
    print(f"------- row {index+1} -------")
    text = tf.expand_dims(row["text"], axis=0)
    preds = model_1.predict_step(text)
    print(f"preds before squeeze/round: {preds}")
    preds = tf.squeeze(preds)  # tf.squeeze(tf.round(preds))
    print(f"Text: {row['text']}")
    print(f"Prediction (n dim array, n=# classes): {preds}")
    preds_np = preds.numpy()
    index_prediction = np.argmax(preds_np)
    print(f"prediction using largest index: {index_prediction}, "
          f"confidence: {preds_np[index_prediction]}")  # {preds[index_prediction]}
    print(f"Ground Truth : {row['target']}")
    if preds[index_prediction] <= 0.5:
        print("WARNING: LOW CONFIDENCE")
    if row['target'] == index_prediction:
        print("CORRECT")
        score += 1
    else:
        print("INCORRECT")
    if index == 10:
        break

print(f"-=-=-=-=- SCORE -=-=-=-=-    =   {float(score/test_records)}")
