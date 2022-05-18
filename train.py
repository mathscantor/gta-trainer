import pandas as pd
from keras import Input
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.layers import Convolution1D, GlobalMaxPooling1D, Concatenate
from keras.models import Model
from keras.metrics import AUC
from utils.converter import Converter
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


def create_rcnn(
    num_classes: int = 2,
    corpus_size: int = 10000,
    input_name: str = "qname",
    bidir: bool = False
) -> Model:
    x = Input(shape=(256,), dtype="int32", name=input_name)
    em1 = Embedding(corpus_size + 1, 128, input_length=256)(x)

    conv1 = Convolution1D(filters=64, kernel_size=2)(em1)
    conv2 = Convolution1D(filters=64, kernel_size=3)(em1)
    conv3 = Convolution1D(filters=64, kernel_size=4)(em1)
    conv4 = Convolution1D(filters=64, kernel_size=5)(em1)
    conv5 = Convolution1D(filters=64, kernel_size=6)(em1)

    pool1 = GlobalMaxPooling1D()(conv1)
    pool2 = GlobalMaxPooling1D()(conv2)
    pool3 = GlobalMaxPooling1D()(conv3)
    pool4 = GlobalMaxPooling1D()(conv4)
    pool5 = GlobalMaxPooling1D()(conv5)
    conv = Concatenate(axis=-1)([pool1, pool2, pool3, pool4, pool5])

    conv = Dropout(0.5)(conv)
    conv = Dense(64, activation="relu")(conv)
    conv = Dropout(0.5)(conv)

    lstm = Bidirectional(LSTM(128))(em1) if bidir else LSTM(128)(em1)

    lstm = Dropout(0.5)(lstm)
    lstm = Dense(128)(lstm)

    input = Concatenate(axis=-1)([conv, lstm])
    input = Dropout(0.5)(input)

    h = Dense(128, activation="relu", kernel_initializer="lecun_normal")(input)
    h = Dropout(0.5)(h)
    y = Dense(num_classes, activation="softmax")(h)

    model = Model(x, y)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", AUC()]
    )
    return model




def main():

    df = pd.read_csv("./datasets/gta-v2.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    domains = df["domain"].values
    labels = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(domains, labels, test_size=0.2, random_state=42)
    print("Getting Training Batches ready...")
    x_train_tensor = converter.convert_domains_to_encoded_tensor(x_train)
    y_train_tensor = converter.convert_labels_to_tensor(y_train)
    print("Getting Validation Batches ready...")
    x_test_tensor = converter.convert_domains_to_encoded_tensor(x_test)
    y_test_tensor = converter.convert_labels_to_tensor(y_test)

    model = create_rcnn(num_classes=2, corpus_size=10000, input_name="qname", bidir=True)
    filepath = "./rcnn/gta-v2-{epoch:02d}-{val_accuracy:.3f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', save_freq="epoch")
    callbacks_list = [checkpoint]
    model.fit(x_train_tensor, y_train_tensor, epochs=10, verbose=1, validation_data=(x_test_tensor,y_test_tensor), callbacks=callbacks_list)




if __name__ == '__main__':

    converter = Converter(max_characters=256)
    main()

# REDACTED CODE
'''iteration = 1
    batch_size = 128
    for lower in range(0, len(domains), batch_size):
        upper = lower + batch_size
        if upper <= len(domains):
            domains_batch = domains[lower:upper]
            labels_batch = labels[lower:upper]
        else:
            domains_batch = domains[lower:len(domains)]
            labels_batch = labels[lower:len(domains)]'''
