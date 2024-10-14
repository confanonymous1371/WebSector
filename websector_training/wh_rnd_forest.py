import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, hamming_loss, accuracy_score, multilabel_confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import gc

# File paths
directory = "/data/sxs7285/Porjects_code/sector_activity/data/data_csv2/"
output_dir_base = "/data/sxs7285/Porjects_code/sector_activity/SPL_classification/results/summary_rnd_forest_wan_sum_new"
os.makedirs(output_dir_base, exist_ok=True)

def save_to_file(content, file_path):
    with open(file_path, 'w') as f:
        f.write(content)

# Load and preprocess data
def load_and_preprocess_data(file_path, chunk_size=10000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    filtered_chunks = []
    for chunk in chunks:
        filtered_chunk = chunk[chunk['content_word_count'] >= chunk['content_word_count'].quantile(0.15)]
        filtered_chunk = filtered_chunk[['content', 'extractive_summary','label_coarse']]
        filtered_chunks.append(filtered_chunk)
    return pd.concat(filtered_chunks)

# WAN loss function
def wan_loss(y_true, y_pred, gamma):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    logits = np.log(y_pred / (1 - y_pred))
    
    observed_labels = y_true.copy()
    loss_mtx = np.zeros_like(observed_labels, dtype=float)
    loss_mtx[observed_labels == 1] = -np.log(1 / (1 + np.exp(-logits[observed_labels == 1])) + 1e-5)
    loss_mtx[observed_labels == 0] = gamma * -np.log(1.0 - 1 / (1 + np.exp(-logits[observed_labels == 0])) + 1e-5)
    return np.mean(loss_mtx)

# Load and preprocess data
data_file = os.path.join(directory, "Linekdin_dataset_Finale.csv")
filtered_linkedin = load_and_preprocess_data(data_file)
print("filtered_linkedin",filtered_linkedin)
# Prepare data for modeling
X = filtered_linkedin['extractive_summary'].values
# X = filtered_linkedin['content'].values

y = filtered_linkedin['label_coarse']
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y.explode())
y = np.eye(len(label_encoder.classes_))[encoded_labels].reshape(len(y), -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000, dtype=np.float32)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# for gamma in np.linspace(0.0, 1.0, 11):
for gamma in [0.1, 0.3]:

    output_dir = f"{output_dir_base}{str(gamma)}/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a custom scorer
    custom_scorer = make_scorer(wan_loss, needs_proba=True, greater_is_better=False, gamma=gamma)

    # Train model with cross-validation using WAN loss
    base_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model = MultiOutputClassifier(base_model)

    # Train the final model on the full training set
    model.fit(X_train, y_train)

    # Get predictions on the test set
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (np.array([proba[:, 1] for proba in y_pred_proba]).T > 0.5).astype(int)

    # Compute WAN loss on test set
    test_loss = wan_loss(y_test, np.array([proba[:, 1] for proba in y_pred_proba]).T, gamma)
    print(f"WAN loss on test set for gamma={gamma}: {test_loss}")

    # Print and save classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    save_to_file(report, os.path.join(output_dir, 'classification_report.txt'))

    # Compute and save metrics
    metrics = {
        "WAN Loss": test_loss,
        "Hamming Loss": hamming_loss(y_test, y_pred),
        "Mean Accuracy": accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
    }

    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    # Generate confusion matrix for each label
    conf_matrix = multilabel_confusion_matrix(y_test, y_pred)
    save_to_file(f"Confusion Matrix (Multi-label):\n{conf_matrix}", os.path.join(output_dir, 'confusion_matrix.txt'))

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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

    # Save true labels and predictions
    np.savetxt(os.path.join(output_dir, 'y_true.txt'), y_test)
    np.savetxt(os.path.join(output_dir, 'y_true.csv'), y_test, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'probs.txt'), np.array([proba[:, 1] for proba in y_pred_proba]).T)
    np.savetxt(os.path.join(output_dir, 'probs.csv'), np.array([proba[:, 1] for proba in y_pred_proba]).T, delimiter=',')

    print(f"Results saved in {output_dir}")
    
    # Release memory
    del model, y_pred_proba, y_pred, conf_matrix, report, metrics
    gc.collect()
