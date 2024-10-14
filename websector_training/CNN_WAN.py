import os
import pandas as pd
import numpy as np
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download NLTK punkt tokenizer
nltk.download('punkt', quiet=True)

# Define directories
directory = "/data/sxs7285/Porjects_code/sector_activity/data/data_csv2/"
glove_path = '/data/sxs7285/Porjects_code/Linkedin/summary_summer/glove/glove.6B.300d.txt'

# Load GloVe embeddings
def load_glove_model(glove_path):
    print("Loading GloVe embeddings...")
    glove_model = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.asarray(parts[1:], dtype='float32')
            glove_model[word] = vector
    print(f"Loaded {len(glove_model)} word vectors.")
    return glove_model

glove_model = load_glove_model(glove_path)

# Data preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def text_to_embedding(text, model, embedding_dim=300):
    words = nltk.word_tokenize(text)
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros((embedding_dim,))
    return np.mean(word_vectors, axis=0)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# CNN Model
class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(256 * (embedding_dim // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return logits

# WAN Loss
class WANLoss(nn.Module):
    def __init__(self, gamma=0.3):
        super(WANLoss, self).__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        observed_labels = labels.clone()
        loss_mtx = torch.zeros_like(observed_labels)
        
        # Positive samples
        positive_mask = observed_labels == 1
        loss_mtx[positive_mask] = -torch.log(torch.sigmoid(logits[positive_mask]) + 1e-5)
        
        # Negative samples
        negative_mask = observed_labels == 0
        loss_mtx[negative_mask] = self.gamma * -torch.log(1.0 - torch.sigmoid(logits[negative_mask]) + 1e-5)
        
        return loss_mtx.mean()

# Metrics calculation functions
def compute_tp_fp_fn_tn(targs, preds):
    tp = np.sum((targs == 1) & (preds >= 0.5), axis=0)
    fp = np.sum((targs == 0) & (preds >= 0.5), axis=0)
    fn = np.sum((targs == 1) & (preds < 0.5), axis=0)
    tn = np.sum((targs == 0) & (preds < 0.5), axis=0)
    return tp, fp, fn, tn

def accuracy(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    return np.sum(tp + tn) / np.sum(tp + fp + fn + tn)

def precision_micro(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    return np.sum(tp) / (np.sum(tp) + np.sum(fp))

def precision_macro(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    precisions = tp / (tp + fp)
    return np.mean(precisions)

def precision_weighted(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    precisions = tp / (tp + fp)
    weights = np.sum(targs, axis=0) / np.sum(targs)
    return np.sum(precisions * weights)

def recall_micro(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    return np.sum(tp) / (np.sum(tp) + np.sum(fn))

def recall_macro(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    recalls = tp / (tp + fn)
    return np.mean(recalls)

def recall_weighted(targs, preds):
    tp, fp, fn, tn = compute_tp_fp_fn_tn(targs, preds)
    recalls = tp / (tp + fn)
    weights = np.sum(targs, axis=0) / np.sum(targs)
    return np.sum(recalls * weights)


def top_n_recall(targs, preds, n):
    sorted_indices = np.argsort(preds, axis=1)[:, ::-1]
    top_n_preds = np.zeros_like(preds)
    for i, indices in enumerate(sorted_indices[:, :n]):
        mask = preds[i, indices] > 0.5
        top_n_preds[i, indices[mask]] = 1
    tp = np.sum(targs * top_n_preds, axis=1)
    return np.mean(tp)
 

def auc(targs, preds):
    return roc_auc_score(targs, preds, average='macro')

def prediction_entropy(preds):
    return -np.sum(preds * np.log(preds + 1e-10), axis=1).mean()

def compute_all_metrics(targs, preds):
    return {
        "Accuracy": accuracy(targs, preds),
        "Precision (Micro)": precision_micro(targs, preds),
        "Precision (Macro)": precision_macro(targs, preds),
        "Precision (Weighted)": precision_weighted(targs, preds),
        "Recall (Micro)": recall_micro(targs, preds),
        "Recall (Macro)": recall_macro(targs, preds),
        "Recall (Weighted)": recall_weighted(targs, preds),
        "Top-2 Recall": top_n_recall(targs, preds, 2),
        "Top-3 Recall": top_n_recall(targs, preds, 3),
        "Top-4 Recall": top_n_recall(targs, preds, 4),
        "AUC": auc(targs, preds),
        "Prediction Entropy": prediction_entropy(preds)
    }

# Training and evaluation function
def train_and_evaluate(train_loader, val_loader, num_classes, gamma, result_directory):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNModel(embedding_dim=300, num_classes=num_classes).to(device)
    criterion = WANLoss(gamma=gamma)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    num_epochs = 30
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Gamma: {gamma}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Check if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()

    # Load the best model
    model.load_state_dict(best_model)
    return model

# Function to get predictions
def get_predictions(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            logits = model(inputs)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
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

    filtered_linkedin['cleaned_summary'] = filtered_linkedin['extractive_summary'].apply(clean_text)
    filtered_linkedin['embedding'] = filtered_linkedin['cleaned_summary'].apply(lambda x: text_to_embedding(x, glove_model, 300))

    X = np.vstack(filtered_linkedin['embedding'].values)

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()

    # Assuming 'filtered_linkedin' is your DataFrame and 'label_coarse' is the column with categorical data
    encoded_labels = label_encoder.fit_transform(filtered_linkedin['label_coarse'])

    # Create one-hot encoded labels
    y = np.eye(len(label_encoder.classes_))[encoded_labels]

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Create DataLoaders
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    test_dataset = TextDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Loop over different gamma values
    gamma_values = np.linspace(0.0, 1.0, 11)
    all_metrics = {}

    for gamma in gamma_values:
        print(f"\nTraining and evaluating model with gamma={gamma}")
        result_directory = f"/data/sxs7285/Porjects_code/sector_activity/SPL_classification/results/CNN_WAN_{str(gamma)}/"
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        
        # Train the model
        model = train_and_evaluate(train_loader, val_loader, num_classes=y.shape[1], gamma=gamma, result_directory=result_directory)

        # Get predictions on the test set
        y_true, probs = get_predictions(model, test_loader, device)
        preds = (probs > 0.5).astype(int)

        # Print and save classification report
        report = classification_report(y_true, preds, target_names=label_encoder.classes_, zero_division=0)
        with open(os.path.join(result_directory, 'report.txt'), 'w') as f:
            f.write("Classification Report:\n")
            f.write(report)

        # Compute and save metrics
        test_metrics = compute_all_metrics(y_true, probs)
        all_metrics[gamma] = test_metrics
        
        with open(os.path.join(result_directory, 'test_results.txt'), 'w') as f:
            for key, value in test_metrics.items():
                f.write(f"{key}: {value}\n")

        # Save true labels and predictions
        np.savetxt(os.path.join(result_directory, 'y_true.txt'), y_true)
        np.savetxt(os.path.join(result_directory, 'y_true.csv'), y_true, delimiter=',')
        np.savetxt(os.path.join(result_directory, 'probs.txt'), probs)
        np.savetxt(os.path.join(result_directory, 'probs.csv'), probs, delimiter=',')
    
    # Save all metrics for comparison
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(directory, 'all_metrics.csv'))
    print("All metrics saved.")
