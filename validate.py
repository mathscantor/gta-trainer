from keras.models import load_model
import pandas as pd
from utils.metrics import Metrics
from utils.converter import Converter
import tensorflow as tf


def main():
    model = load_model("./rcnn/gta-v2-04-0.994.h5")
    df = pd.read_csv("./datasets/gta-test.csv")
    domains = df["domain"].values
    labels = df["label"].values

    i = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for domain in domains:
        domain_encoded_string_tensor = converter.convert_domain_to_encoded_tensor(domain)

        if model.predict(domain_encoded_string_tensor.numpy())[0][1] > 0.5:
            if labels[i] != "legit" or labels[i] != "normal":
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == "legit":
                tn += 1
            else:
                fn += 1
        i += 1

    results = metrics.confusion_matrix(tp, tn, fp, fn)
    print("Confusion Matrix >> tpr: {}, tnr: {}, fpr: {}, fnr: {} ".format(results[0], results[1], results[2], results[3]))
    results = metrics.accuracy(tp, tn, fp, fn)
    print("Accuracy: {}".format(results))
    results = metrics.f1_score(tp,fp,fn)
    print("F1 score: {}".format(results))

if __name__ == '__main__':
    converter = Converter(max_characters=256)
    metrics = Metrics()
    main()

