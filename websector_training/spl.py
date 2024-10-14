
import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score



loss = "wan"

gamma=0.3                       # gamma value for WAN loss
ls_coef=0.1                    # label smoothing coefficient
expected_num_pos=2              # expected number of positives for EPR and ROLE loss

os.environ["WANDB_PROJECT"] = "PrivaSeer"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

directory = "/data/sxs7285/Porjects_code/sector_activity/data/data_csv2/"
linkedin = pd.read_csv(os.path.join(directory, "Linekdin_dataset_Finale.csv"))

def cumalative_freq(data, word_count):
    quantiles = data[word_count].quantile([0.15,0.25,0.5,0.75])
    total_samples = len(data[word_count])
    print(f"Total samples: {total_samples}")
    for quantile in quantiles.index:
        count_above = (data[word_count] > quantiles[quantile]).sum()
        print(f"Number of samples above the {quantile*100}% quantile ({quantiles[quantile]}): {count_above}")

cumalative_freq(linkedin, "content_word_count")
cumalative_freq(linkedin, "summary_word_count")

def filter_by_word_count_quantile(df, word_count_column, quantile_threshold):
    quantile_value = df[word_count_column].quantile(quantile_threshold)
    filtered_df = df[df[word_count_column] >= quantile_value]
    return filtered_df

filtered_linkedin = filter_by_word_count_quantile(linkedin, 'content_word_count', 0.15)

# filtered_linkedin = filtered_linkedin.sample(1000)

print(f"Original dataset size: {len(linkedin)}")
print(f"Filtered dataset size: {len(filtered_linkedin)}")

class_column = 'label_coarse'
train_data_cv, test_data = train_test_split(filtered_linkedin, test_size=0.3, random_state=42, stratify=filtered_linkedin[class_column])
train_data, val_data = train_test_split(train_data_cv, test_size=0.2, random_state=42, stratify=train_data_cv[class_column])

# Encode labels
label_encoder = LabelEncoder()
train_data['label_fine'] = train_data['label']
train_data['labels'] = label_encoder.fit_transform(train_data['label_coarse'])  # Rename to 'labels'
train_data['labels'] = list(np.eye(10)[train_data['labels']])

val_data['label_fine'] = val_data['label']
val_data['labels'] = label_encoder.fit_transform(val_data['label_coarse'])  # Rename to 'labels'
val_data['labels'] = list(np.eye(10)[val_data['labels']])

test_data['label_fine'] = test_data['label']
test_data['labels'] = label_encoder.fit_transform(test_data['label_coarse'])  # Rename to 'labels'
test_data['labels'] = list(np.eye(10)[test_data['labels']])


from transformers import BertTokenizer, BertForSequenceClassification

# # Define the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
# model.to(device)


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples['extractive_summary'], truncation=True, padding=True)

train_dataset = Dataset.from_pandas(train_data[['extractive_summary', 'labels']])
val_dataset = Dataset.from_pandas(val_data[['extractive_summary', 'labels']])
test_dataset = Dataset.from_pandas(test_data[['extractive_summary', 'labels']])

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

import numpy as np
from sklearn.metrics import average_precision_score

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
from scipy.special import xlogy

# def check_inputs(targs, preds):
#     assert (np.shape(preds) == np.shape(targs))
#     assert isinstance(preds, np.ndarray)
#     assert isinstance(targs, np.ndarray)
#     assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
#     assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
#     assert (np.sum(targs, axis=1) == 1).all()  # Ensure only one positive label per sample

# def compute_recall(targs, preds):
#     check_inputs(targs, preds)
#     tp = np.sum((targs == 1) & (preds >= 0.5), axis=0)
#     fn = np.sum((targs == 1) & (preds < 0.5), axis=0)
#     recall = np.mean(tp / (tp + fn + 1e-10))
#     return recall

# def compute_precision(targs, preds):
#     check_inputs(targs, preds)
#     tp = np.sum((targs == 1) & (preds >= 0.5), axis=0)
#     fp = np.sum((targs == 0) & (preds >= 0.5), axis=0)
#     precision = np.mean(tp / (tp + fp + 1e-10))
#     return precision

# def compute_ranked_accuracy_at_k(targs, preds, k):
#     check_inputs(targs, preds)
#     accuracies = []
#     for i in range(targs.shape[0]):
#         top_k_pred = np.argsort(preds[i])[::-1][:k]
#         true_label = np.argmax(targs[i])
#         accuracies.append(int(true_label in top_k_pred))
#     return np.mean(accuracies)

# def compute_cumulative_accuracy_at_k(targs, preds, k):
#     cumulative_accuracy = 0
#     for i in range(1, k + 1):
#         cumulative_accuracy += compute_ranked_accuracy_at_k(targs, preds, i)
#     return cumulative_accuracy

# def compute_recall_at_k(targs, preds, k):
#     check_inputs(targs, preds)
#     recalls = []
#     for i in range(targs.shape[0]):
#         top_k_pred = np.argsort(preds[i])[::-1][:k]
#         true_label = np.argmax(targs[i])
#         recalls.append(int(true_label in top_k_pred))
#     return np.mean(recalls)

# # a. Propensity Scored Precision (PSP)
# def compute_psp(targs, preds, propensity_scores):
#     check_inputs(targs, preds)
#     y_pred_binary = (preds >= 0.5).astype(int)
#     psp = np.sum(targs * y_pred_binary / propensity_scores) / np.sum(y_pred_binary)
#     return psp

# # b. Expected Ranking Accuracy
# def compute_era(targs, preds):
#     check_inputs(targs, preds)
#     rankings = np.argsort(-preds, axis=1)
#     true_label_rankings = np.array([np.where(rankings[i] == np.argmax(targs[i]))[0][0] 
#                                     for i in range(len(targs))])
#     return np.mean(1 / (true_label_rankings + 1))

# # c. Partial AUC
# def compute_partial_auc(targs, preds, max_fpr=0.1):
#     check_inputs(targs, preds)
#     fpr, tpr, _ = roc_curve(targs.ravel(), preds.ravel())
#     partial_auc_value = auc(fpr[fpr <= max_fpr], tpr[fpr <= max_fpr])
#     return partial_auc_value / max_fpr

# # d. LDAM-inspired metric
# def compute_ldam_metric(targs, preds):
#     check_inputs(targs, preds)
#     class_counts = np.sum(targs, axis=0)
#     class_weights = 1 / np.sqrt(class_counts + 1e-5)
#     ldam_scores = np.sum(targs * class_weights * np.log(preds + 1e-5), axis=1)
#     return np.mean(ldam_scores)

# # e. Confidence Calibration (Expected Calibration Error)
# def compute_ece(targs, preds, n_bins=10):
#     check_inputs(targs, preds)
#     bin_boundaries = np.linspace(0, 1, n_bins + 1)
#     bin_lowers = bin_boundaries[:-1]
#     bin_uppers = bin_boundaries[1:]
    
#     confidences = np.max(preds, axis=1)
#     predictions = np.argmax(preds, axis=1)
#     accuracies = predictions == np.argmax(targs, axis=1)
    
#     ece = 0
#     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
#         prop_in_bin = np.mean(in_bin)
#         if prop_in_bin > 0:
#             accuracy_in_bin = np.mean(accuracies[in_bin])
#             avg_confidence_in_bin = np.mean(confidences[in_bin])
#             ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
#     return ece

# # f. Entropy of Predictions
# def compute_prediction_entropy(preds):
#     return -np.mean(np.sum(xlogy(preds, preds), axis=1))


# def compute_tp_fp_fn_tn(targs, preds):
#     check_inputs(targs, preds)
#     tp = np.sum((targs == 1) & (preds >= 0.5), axis=0)
#     fp = np.sum((targs == 0) & (preds >= 0.5), axis=0)
#     fn = np.sum((targs == 1) & (preds < 0.5), axis=0)
#     tn = np.sum((targs == 0) & (preds < 0.5), axis=0)
#     return tp, fp, fn, tn

# # Helper function for computing precision at k
# def compute_precision_at_k(targs, preds, k):
#     check_inputs(targs, preds)
#     top_k_preds = np.argsort(preds, axis=1)[:, -k:]
#     precision_at_k = np.mean([np.sum(targs[i, top_k_preds[i]]) / k for i in range(targs.shape[0])])
#     return precision_at_k


import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score

def compute_precision_at_k(labels, preds_probs, k, threshold=0.5):
    # Compute the indices of top k predictions
    top_k_indices = np.argsort(preds_probs, axis=1)[:, -k:]
    
    # Filter predictions by threshold
    top_k_preds_thresholded = preds_probs[np.arange(preds_probs.shape[0])[:, None], top_k_indices] >= threshold
    
    # Calculate correct predictions
    correct = (labels[np.arange(labels.shape[0])[:, None], top_k_indices] == 1) & top_k_preds_thresholded
    precision_at_k = np.mean(correct.sum(axis=1) / k)
    
    return precision_at_k

def compute_recall_at_k(labels, preds_probs, k, threshold=0.5):
    # Compute the indices of top k predictions
    top_k_indices = np.argsort(preds_probs, axis=1)[:, -k:]
    
    # Filter predictions by threshold
    top_k_preds_thresholded = preds_probs[np.arange(preds_probs.shape[0])[:, None], top_k_indices] >= threshold
    
    # Calculate correct predictions
    correct = (labels[np.arange(labels.shape[0])[:, None], top_k_indices] == 1) & top_k_preds_thresholded
    recall_at_k = np.mean(correct.sum(axis=1) / labels.sum(axis=1))
    
    return recall_at_k

def compute_ranked_accuracy_at_k(labels, preds_probs, k, threshold=0.5):
    # Compute the indices of the k-th ranked predictions
    k_ranked_indices = np.argsort(preds_probs, axis=1)[:, -k]
    
    # Check if the k-th ranked predictions are correct and above the threshold
    k_ranked_preds_thresholded = preds_probs[np.arange(preds_probs.shape[0]), k_ranked_indices] >= threshold
    correct = (labels[np.arange(labels.shape[0]), k_ranked_indices] == 1) & k_ranked_preds_thresholded
    
    ranked_accuracy_at_k = np.mean(correct)
    
    return ranked_accuracy_at_k


def compute_era(labels, preds_probs):
    # Dummy implementation for ERA
    return np.random.rand()

def compute_partial_auc(labels, preds_probs):
    auc_scores = [roc_auc_score(labels[:, i], preds_probs[:, i]) for i in range(labels.shape[1])]
    partial_auc = np.mean(auc_scores)
    return partial_auc

def compute_ldam_metric(labels, preds_probs):
    # Dummy implementation for LDAM metric
    return np.random.rand()

def compute_ece(labels, preds_probs):
    # Dummy implementation for ECE
    return np.random.rand()

def compute_prediction_entropy(preds_probs):
    entropy = -np.sum(preds_probs * np.log(preds_probs + 1e-12), axis=1).mean()
    return entropy

def compute_psp(labels, preds_probs, propensity_scores):
    precision_scores = precision_score(labels, preds_probs > 0.5, average=None)
    psp = np.sum(precision_scores / propensity_scores) / len(precision_scores)
    return psp

def compute_cumulative_accuracy_at_k(labels, preds_probs, k):

    cumulative_accuracy_at_k = 0
    for i in range(k):

        cumulative_accuracy_at_k =+ compute_ranked_accuracy_at_k(labels, preds_probs,i)

    return cumulative_accuracy_at_k
def compute_all_negatives_count(preds_probs, threshold=0.5):
    # Count instances where all predicted probabilities are below the threshold
    all_negative = np.all(preds_probs < threshold, axis=1)
    return np.sum(all_negative)






from transformers import Trainer
import torch

# Define the CustomTrainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Determine which loss function to use based on the 'loss_function' parameter
        loss_function = self.args.loss_function
        
        # Print the selected loss function
        print(f"Selected loss function: {loss_function}")

        # Binary Cross-Entropy (BCE) Loss
        if loss_function == 'bce':
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        # Binary Cross-Entropy with Label Smoothing (BCE-LS) Loss
        elif loss_function == 'bce_ls':
            ls_coef = self.args.ls_coef
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = (1.0 - ls_coef) * loss_fn(logits, labels.float()) + \
                   ls_coef * loss_fn(-logits, 1 - labels.float())
        
        # Ignore Unobserved Negatives (IUN) Loss
        elif loss_function == 'iun':
            loss = self.iun_loss(logits, labels)
        
        # Ignore Unobserved (IU) Loss
        elif loss_function == 'iu':
            loss = self.iu_loss(logits, labels)
        
        # Pairwise Ranking (PR) Loss
        elif loss_function == 'pr':
            loss = self.pr_loss(logits, labels)
        
        # Assume Negative (AN) Loss
        elif loss_function == 'an':
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
        
        # Assume Negative with Label Smoothing (AN-LS) Loss
        elif loss_function == 'an_ls':
            ls_coef = self.args.ls_coef
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = (1.0 - ls_coef) * loss_fn(logits, labels.float()) + \
                   ls_coef * loss_fn(-logits, 1 - labels.float())
        
        # Weak Assume Negative (WAN) Loss
        elif loss_function == 'wan':
            gamma = self.args.gamma
            loss = self.wan_loss(logits, labels, gamma)
        
        # Expected Positive Regularizer (EPR) Loss
        elif loss_function == 'epr':
            expected_num_pos = self.args.expected_num_pos
            loss = self.epr_loss(logits, labels, expected_num_pos)
        
        # Regularized Online Label Estimation (ROLE) Loss
        elif loss_function == 'role':
            expected_num_pos = self.args.expected_num_pos
            estimated_labels = inputs.pop("estimated_labels")
            loss = self.role_loss(logits, labels, estimated_labels, expected_num_pos)
        
        # If no valid loss function is specified, default to BCE loss
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss

    # Define helper functions for custom loss computations

    def iun_loss(self, logits, labels):
        # Implement the IUN loss function
        observed_labels = labels.clone()
        true_labels = labels.clone()
        loss_mtx = torch.zeros_like(observed_labels)
        loss_mtx[observed_labels == 1] = -torch.log(torch.sigmoid(logits[observed_labels == 1]) + 1e-5)
        loss_mtx[true_labels == -1] = -torch.log(1.0 - torch.sigmoid(logits[true_labels == -1]) + 1e-5)
        loss = loss_mtx.mean()
        return loss

    def iu_loss(self, logits, labels):
        # Implement the IU loss function
        observed_labels = labels.clone()
        loss_mtx = torch.zeros_like(observed_labels)
        loss_mtx[observed_labels == 1] = -torch.log(torch.sigmoid(logits[observed_labels == 1]) + 1e-5)
        loss_mtx[observed_labels == -1] = -torch.log(1.0 - torch.sigmoid(logits[observed_labels == -1]) + 1e-5)
        loss = loss_mtx.mean()
        return loss

    def pr_loss(self, logits, labels):
        # Implement the PR loss function
        observed_labels = labels.clone()
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        loss_mtx = torch.zeros_like(observed_labels)
        for n in range(batch_size):
            preds_neg = logits[n, :][observed_labels[n, :] == 0]
            for i in range(num_classes):
                if observed_labels[n, i] == 1:
                    loss_mtx[n, i] = torch.sum(torch.clamp(1.0 - logits[n, i] + preds_neg, min=0))
        loss = loss_mtx.mean()
        return loss

    def wan_loss(self, logits, labels, gamma):
        # Implement the WAN loss function
        observed_labels = labels.clone()
        loss_mtx = torch.zeros_like(observed_labels)
        loss_mtx[observed_labels == 1] = -torch.log(torch.sigmoid(logits[observed_labels == 1]) + 1e-5)
        loss_mtx[observed_labels == 0] = gamma * -torch.log(1.0 - torch.sigmoid(logits[observed_labels == 0]) + 1e-5)
        loss = loss_mtx.mean()
        return loss

    def epr_loss(self, logits, labels, expected_num_pos):
        # Implement the EPR loss function
        observed_labels = labels.clone()
        loss_mtx = torch.zeros_like(observed_labels)
        loss_mtx[observed_labels == 1] = -torch.log(torch.sigmoid(logits[observed_labels == 1]) + 1e-5)
        reg_loss = self.expected_positive_regularizer(torch.sigmoid(logits), expected_num_pos, norm='2') / (logits.size(1) ** 2)
        loss = loss_mtx.mean() + reg_loss
        return loss

    def role_loss(self, logits, labels, estimated_labels, expected_num_pos):
        # Implement the ROLE loss function
        observed_labels = labels.clone()
        estimated_labels_detached = estimated_labels.detach()
        preds_detached = torch.sigmoid(logits).detach()

        # (image classifier) compute loss w.r.t. observed positives:
        loss_mtx_pos_1 = torch.zeros_like(observed_labels)
        loss_mtx_pos_1[observed_labels == 1] = -torch.log(torch.sigmoid(logits[observed_labels == 1]) + 1e-5)

        # (image classifier) compute loss w.r.t. label estimator outputs:
        loss_mtx_cross_1 = estimated_labels_detached * -torch.log(torch.sigmoid(logits) + 1e-5) + \
                           (1.0 - estimated_labels_detached) * -torch.log(1.0 - torch.sigmoid(logits) + 1e-5)

        # (image classifier) compute regularizer:
        reg_1 = self.expected_positive_regularizer(torch.sigmoid(logits), expected_num_pos, norm='2') / (logits.size(1) ** 2)

        # (label estimator) compute loss w.r.t. observed positives:
        loss_mtx_pos_2 = torch.zeros_like(observed_labels)
        loss_mtx_pos_2[observed_labels == 1] = -torch.log(estimated_labels[observed_labels == 1] + 1e-5)

        # (label estimator) compute loss w.r.t. image classifier outputs:
        loss_mtx_cross_2 = preds_detached * -torch.log(estimated_labels + 1e-5) + \
                           (1.0 - preds_detached) * -torch.log(1.0 - estimated_labels + 1e-5)

        # (label estimator) compute regularizer:
        reg_2 = self.expected_positive_regularizer(estimated_labels, expected_num_pos, norm='2') / (logits.size(1) ** 2)

        # compute final loss matrix:
        reg_loss = 0.5 * (reg_1 + reg_2)
        loss_mtx = 0.5 * (loss_mtx_pos_1 + loss_mtx_pos_2)
        loss_mtx += 0.5 * (loss_mtx_cross_1 + loss_mtx_cross_2)
        
        loss = loss_mtx.mean() + reg_loss
        return loss

    def expected_positive_regularizer(self, preds, expected_num_pos, norm='2'):
        # Implement the regularizer for expected number of positives
        if norm == '1':
            reg = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
        elif norm == '2':
            reg = (preds.sum(1).mean(0) - expected_num_pos)**2
        else:
            raise NotImplementedError
        return reg


from transformers import TrainingArguments

class CustomTrainingArguments(TrainingArguments):
    def __init__(self, loss_function=None, ls_coef=0.0, gamma=0.0, expected_num_pos=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function
        self.ls_coef = ls_coef
        self.gamma = gamma
        self.expected_num_pos = expected_num_pos




import os
import torch
import numpy as np
from transformers import Trainer, TrainingArguments, RobertaForSequenceClassification, RobertaTokenizer
from datasets import Dataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# Assuming necessary imports and helper functions (compute_precision_at_k, compute_recall_at_k, etc.) are already defined







# # Function to compute metrics
# def compute_metrics(pred, labels):
#     labels = pred.label_ids
#     print("pred.predictions")
#     preds_probs = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    
#     precision_at_1 = compute_precision_at_k(labels, preds_probs, k=1)
#     recall_at_1 = compute_recall_at_k(labels, preds_probs, k=1)
#     accuracy_at_1 = compute_ranked_accuracy_at_k(labels, preds_probs, k=1)

#     precision_at_2 = compute_precision_at_k(labels, preds_probs, k=2)
#     recall_at_2 = compute_recall_at_k(labels, preds_probs, k=2)
#     accuracy_at_2 = compute_ranked_accuracy_at_k(labels, preds_probs, k=2)

#     precision_at_3 = compute_precision_at_k(labels, preds_probs, k=3)
#     recall_at_3 = compute_recall_at_k(labels, preds_probs, k=3)
#     accuracy_at_3 = compute_ranked_accuracy_at_k(labels, preds_probs, k=3)

#     accuracy_at_10 = compute_ranked_accuracy_at_k(labels, preds_probs, k=10)
    
#     era = compute_era(labels, preds_probs)
#     partial_auc = compute_partial_auc(labels, preds_probs)
#     ldam_metric = compute_ldam_metric(labels, preds_probs)
#     ece = compute_ece(labels, preds_probs)
#     prediction_entropy = compute_prediction_entropy(preds_probs)
    
#     propensity_scores = np.clip(np.mean(labels, axis=0), 0.1, 0.9)
#     psp = compute_psp(labels, preds_probs, propensity_scores)

#     compute_cumulative_accuracy_at_1 = compute_cumulative_accuracy_at_k(labels, preds_probs, k = 1)
#     compute_cumulative_accuracy_at_2 = compute_cumulative_accuracy_at_k(labels, preds_probs, k =2 )
#     compute_cumulative_accuracy_at_3 = compute_cumulative_accuracy_at_k(labels, preds_probs, k= 3)
    
    

#     all_negatives_count = compute_all_negatives_count(preds_probs)
#     metrics = {
#         # 'subdir': f"loss_wan{subdir}_roberta",
#         'precision@1': precision_at_1,
#         'recall@1': recall_at_1,
#         'accuracy@1': accuracy_at_1,

#         'precision@2': precision_at_2,
#         'recall@2': recall_at_2,
#         'accuracy@2': accuracy_at_2,

#         'precision@3': precision_at_3,
#         'recall@3': recall_at_3,
#         'accuracy@3': accuracy_at_3,

#         'accuracy@10': accuracy_at_10,
#         'cumulative_accuracy@1': compute_cumulative_accuracy_at_1,
#         'cumulative_accuracy@2': compute_cumulative_accuracy_at_2,
#         'cumulative_accuracy@3': compute_cumulative_accuracy_at_3,

#         'era': era,
#         'partial_auc': partial_auc,
#         'ldam_metric': ldam_metric,
#         'ece': ece,
#         'prediction_entropy': prediction_entropy,
#         'all_negatives_count': all_negatives_count,

#         'psp': psp,
#     }
    
#     return metrics


import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from scipy.special import xlogy

def compute_metrics(pred):
    labels = pred.label_ids
    print("pred.predictions")
    preds_probs = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    
    # Existing metrics
    precision_at_1 = compute_precision_at_k(labels, preds_probs, k=1)
    recall_at_1 = compute_recall_at_k(labels, preds_probs, k=1)
    accuracy_at_1 = compute_ranked_accuracy_at_k(labels, preds_probs, k=1)

    precision_at_2 = compute_precision_at_k(labels, preds_probs, k=2)
    recall_at_2 = compute_recall_at_k(labels, preds_probs, k=2)
    accuracy_at_2 = compute_ranked_accuracy_at_k(labels, preds_probs, k=2)


    precision_at_3 = compute_precision_at_k(labels, preds_probs, k=3)
    recall_at_3 = compute_recall_at_k(labels, preds_probs, k=3)
    accuracy_at_3 = compute_ranked_accuracy_at_k(labels, preds_probs, k=3)
    
    # New metrics
    era = compute_era(labels, preds_probs)
    partial_auc = compute_partial_auc(labels, preds_probs)
    ldam_metric = compute_ldam_metric(labels, preds_probs)
    ece = compute_ece(labels, preds_probs)
    prediction_entropy = compute_prediction_entropy(preds_probs)
    
    # Propensity Scored Precision (PSP)
    # Note: You'll need to provide or estimate propensity scores
    # This is a placeholder estimation; replace with your actual propensity score calculation
    propensity_scores = np.clip(np.mean(labels, axis=0), 0.1, 0.9)
    psp = compute_psp(labels, preds_probs, propensity_scores)
    
    # Cumulative Accuracy
    cumulative_accuracy_3 = compute_cumulative_accuracy_at_k(labels, preds_probs, k=3)
    cumulative_accuracy_2 = compute_cumulative_accuracy_at_k(labels, preds_probs, k=2)

    
    metrics = {
        'precision@1': precision_at_1,
        'recall@1': recall_at_1,
        'accuracy@1': accuracy_at_1,

        'precision@2': precision_at_2,
        'recall@2': recall_at_2,
        'accuracy@2': accuracy_at_2,
        'cumulative_accuracy@2': cumulative_accuracy_2,

        'precision@3': precision_at_3,
        'recall@3': recall_at_3,
        'accuracy@3': accuracy_at_3,
        'cumulative_accuracy@3': cumulative_accuracy_3,
        # New metrics
        'era': era,
        'partial_auc': partial_auc,
        'ldam_metric': ldam_metric,
        'ece': ece,
        'prediction_entropy': prediction_entropy,
        'psp': psp,
    }
    
    return metrics

# Note: The following functions are assumed to be defined elsewhere:
# compute_precision_at_k, compute_recall_at_k, compute_avg_precision,
# compute_weighted_precision_recall





# Define the output directory
output_dir = f'/data/sxs7285/Porjects_code/sector_activity/SPL_classification/results/loss_{loss}{str(gamma)}_nol_Roberta/'
os.makedirs(output_dir, exist_ok=True)


training_args = CustomTrainingArguments(
    output_dir=os.path.join(output_dir, './results'),
    num_train_epochs=7,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy='epoch',
    eval_steps=100,
    learning_rate=5e-6,
    save_strategy='epoch',
    logging_strategy="steps",
    logging_dir=os.path.join(output_dir, './logs'),
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1,  # Only keep the best model
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    weight_decay=0.1,
    # report_to="wandb",
    loss_function=loss,
    ls_coef=ls_coef,
    gamma=gamma,
    expected_num_pos=expected_num_pos,
)


from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import get_peft_model, LoraConfig
import torch

###########################

# You may wabt to alter lora_dropout=0.6
# With lora_dropout=0.1

#######################

peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=128,
    lora_alpha=512,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)

# # Initialize tokenizer and base model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))

# from transformers import BertTokenizer, BertForSequenceClassification

# # Define the tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# base_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# # Apply PEFT to the model
# model = get_peft_model(base_model, peft_config)
# model.print_trainable_parameters()  # This will show the number of trainable parameters

# Move the model to the appropriate device
model.to(device)

# Define data_collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the best model
best_model_path = os.path.join(output_dir, 'best_model')
trainer.save_model(best_model_path)

# Save the PEFT configuration
peft_config.save_pretrained(best_model_path)

print(f"Best model saved to {best_model_path}")



# Evaluate the model
eval_result = trainer.evaluate()
print(eval_result)



with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
    for key, value in eval_result.items():
        f.write(f"{key}: {value}\n")

# Get predictions on the test set
predictions = trainer.predict(test_dataset)
preds = torch.sigmoid(torch.tensor(predictions.predictions)) > 0.5
preds = preds.cpu().numpy()
probs = torch.sigmoid(torch.tensor(predictions.predictions)).cpu().numpy()

# Get true labels
y_true = np.array(list(test_data['labels']))

# Print classification report
report = classification_report(y_true, preds, target_names=label_encoder.classes_, zero_division=0)
print(report)

# Save confusion matrix to a text file
with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
    f.write("report:\n")
    f.write(report)

# Compute and save metrics
test_metrics = compute_metrics(predictions)
print(test_metrics)



with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
    for key, value in test_metrics.items():
        f.write(f"{key}: {value}\n")

# Generate confusion matrix for each label
conf_matrix = multilabel_confusion_matrix(y_true, preds)
print("Confusion Matrix (Multi-label):\n", conf_matrix)

with open(os.path.join(output_dir, 'confusion_matrix.txt'), 'w') as f:
    f.write("Confusion Matrix (Multi-label):\n")
    f.write(np.array2string(conf_matrix))

# Normalize the confusion matrix row-wise
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=2)[:, :, np.newaxis]

# Visualization for each class
fig, axes = plt.subplots(len(label_encoder.classes_) + 1, 1, figsize=(10, 5 * (len(label_encoder.classes_) + 1)))
for i, class_name in enumerate(label_encoder.classes_):
    tn, fp, fn, tp = conf_matrix[i].ravel()
    bars = axes[i].bar(['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                       [tn, fp, fn, tp],
                       color=['green', 'blue', 'red', 'grey'])
    axes[i].set_title(f'Confusion Matrix for: {class_name}')
    axes[i].set_ylabel('Count')
    axes[i].set_ylim(0, np.max(conf_matrix[i]))
    for bar in bars:
        yval = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

    fig_class_path = os.path.join(output_dir, f'confusion_matrix_{class_name}.png')
    fig.savefig(fig_class_path)

# Overall visualization
total_conf_matrix = conf_matrix.sum(axis=0)
tn, fp, fn, tp = total_conf_matrix.ravel()
bars = axes[-1].bar(['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                    [tn, fp, fn, tp],
                    color=['green', 'blue', 'red', 'grey'])
axes[-1].set_title('Overall Confusion Matrix')
axes[-1].set_ylabel('Count')
axes[-1].set_ylim(0, np.max(total_conf_matrix))
for bar in bars:
    yval = bar.get_height()
    axes[-1].text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

fig_overall_path = os.path.join(output_dir, 'overall_confusion_matrix.png')
fig.savefig(fig_overall_path)

plt.tight_layout()
plt.show()

# Save true labels and predictions
np.savetxt(os.path.join(output_dir,'y_true.txt'), y_true)
np.savetxt(os.path.join(output_dir,'y_true.csv'), y_true, delimiter=',')
np.savetxt(os.path.join(output_dir,'probs.txt'), probs)
np.savetxt(os.path.join(output_dir,'probs.csv'), probs, delimiter=',')
