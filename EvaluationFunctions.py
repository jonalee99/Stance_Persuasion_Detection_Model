import datetime
import os

import numpy as np

num_labels = 2

NN_RESULTS_PATH = os.path.join("logs", str(datetime.datetime.now().time()))


def get_label_measures(y_true, y_pred):

    y_pred_arr = y_pred.numpy()
    y_true_arr = y_true.numpy()
    predictions = np.zeros([len(y_pred_arr), 1])

    for i in range(0, len(y_pred_arr)):

        predictions[i] = np.argmax(y_pred_arr[i])

    label_measures = []
    for value in range(num_labels):

        label_measures.append(dict())

    for label in range(num_labels):
        for index, correct_results in enumerate(y_true_arr):

            correct_results = y_true_arr[index]

            if correct_results[label] == 1 and predictions[index] == label:
                if 'tp' in label_measures[label]:
                    label_measures[label]['tp'] += 1
                else:
                    label_measures[label]['tp'] = 1
            elif correct_results[label] == 1:
                if 'fn' in label_measures[label]:
                    label_measures[label]['fn'] += 1
                else:
                    label_measures[label]['fn'] = 1
            elif correct_results[label] != 1 and predictions[index] != label:
                if 'tn' in label_measures[label]:
                    label_measures[label]['tn'] + 1
                else:
                    label_measures[label]['tn'] = 1
            elif correct_results[label] != 1 and predictions[index] == label:
                if 'fp' in label_measures[label]:
                    label_measures[label]['fp'] += 1
                else:
                    label_measures[label]['fp'] = 1

    return label_measures


def get_tp_fp_fn(label_measures):
    tps = []
    fns = []
    fps = []
    tns = []
    for label in range(num_labels):
        try:
            tp = label_measures[label]['tp']
        except KeyError:
            tp = 0
        try:
            fp = label_measures[label]['fp']
        except KeyError:
            fp = 0
        try:
            fn = label_measures[label]['fn']
        except KeyError:
            fn = 0
        try:
            tn = label_measures[label]['tn']
        except KeyError:
            tn = 0
        fps.append(fp)
        tps.append(tp)
        fns.append(fn)
        tns.append(tn)
    return tps, fps, fns, tns


def recall(y_true, y_pred):
    label_measures = get_label_measures(y_true, y_pred)
    tps, _, fns, _ = get_tp_fp_fn(label_measures)
    # Return micro-average
    return sum(tps) / (sum(tps) + sum(fns))


def precision(y_true, y_pred):
    label_measures = get_label_measures(y_true, y_pred)
    tps, fps, _, _ = get_tp_fp_fn(label_measures)
    # Return the micro-average
    return sum(tps) / (sum(tps) + sum(fps))


def f1(y_true, y_pred):
    micro_p = precision(y_true, y_pred)
    micro_r = recall(y_true, y_pred)
    if (micro_p + micro_r) == 0:
        micro_p += 0.0001
    return 2 * ((micro_p * micro_r) / (micro_p + micro_r))


def write_metrics(y_true, y_pred):
    label_measures = get_label_measures(y_true, y_pred)
    tps, fps, fns, tns = get_tp_fp_fn(label_measures)
    f1s = []
    recalls = []
    precisions = []
    for label in range(num_labels):
        tp = tps[label]
        fn = fns[label]
        fp = fps[label]
        if tp != 0 or (fn != 0 and fp != 0):
            f1 = tp / (tp + 0.5 * (fp + fn))
            f1s.append(f1)
        if tp != 0 or fn != 0:
            recall = float(tp) / (float(tp) + float(fn))
            recalls.append(recall)
        if tp != 0 or fp != 0:
            precision = float(tp) / (float(tp) + float(fp))
            precisions.append(precision)

    headers = ["Precisions", "Recalls", "F1s"]
    results = [precisions, recalls, f1s]
    if not os.path.exists(NN_RESULTS_PATH):
        os.makedirs(NN_RESULTS_PATH)

    with open(NN_RESULTS_PATH + '/metrics-{}.txt'.format(datetime.datetime.now().time()), 'a+') as file:

        for index, header in enumerate(headers):
            # Write label specific metrics to file
            file.write(header + "\n")
            file.write(",".join([str(l2) for l2 in results[index]]))
            file.write("\n")
        for label in range(num_labels):
            file.write("Confusion Matrix for label {}".format(str(label)))
            file.write("True Positives: {}".format(str(tps[label])))
            file.write("False Negatives: {}".format(str(fns[label])))
            file.write("False Positives: {}".format(str(fps[label])))
            file.write("True Negatives: {}".format(str(tns[label])))
    return 0