import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


PATIENT_DATA_CSV_PATH = '/content/patient_severity_dataset.csv'
patient_df = pd.read_csv(PATIENT_DATA_CSV_PATH)

patient_df = patient_df.dropna(subset=['Severity Level', 'Symptoms & Clinical Features'])


print(f"Original severity levels: {patient_df['Severity Level'].unique()}")
patient_df['Severity Level'] = patient_df['Severity Level'].str.lower().str.strip()
print(f"Standardized severity levels: {patient_df['Severity Level'].unique()}")

patient_df = patient_df[patient_df['Severity Level'] != 'moderate']
print(f"Severity levels after filtering: {patient_df['Severity Level'].unique()}")
print(f"Patient dataset shape after filtering moderate: {patient_df.shape}")


severity_mapping = {"mild": 0, "severe": 1, "critical": 2}


patient_df['Severity Level'] = patient_df['Severity Level'].map(severity_mapping)
print(f"Count of NaN severity levels: {patient_df['Severity Level'].isna().sum()}")


patient_df = patient_df.dropna(subset=['Severity Level'])
print(f"Patient dataset shape after removing NaN severity levels: {patient_df.shape}")


train_texts_patient, test_texts_patient, train_labels_patient, test_labels_patient = train_test_split(
    patient_df['Symptoms & Clinical Features'].tolist(),
    patient_df['Severity Level'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=patient_df['Severity Level']
)

train_texts_patient, val_texts_patient, train_labels_patient, val_labels_patient = train_test_split(
    train_texts_patient,
    train_labels_patient,
    test_size=0.125,
    random_state=42,
    stratify=train_labels_patient
)


class PatientDataset(Dataset):
    def _init_(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None  
        )

        
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)

        return item


MODEL_NAME = "dmis-lab/biobert-v1.1"
tokenizer_patient = AutoTokenizer.from_pretrained(MODEL_NAME)
model_patient = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)


train_dataset_patient = PatientDataset(train_texts_patient, train_labels_patient, tokenizer_patient)
val_dataset_patient = PatientDataset(val_texts_patient, val_labels_patient, tokenizer_patient)
test_dataset_patient = PatientDataset(test_texts_patient, test_labels_patient, tokenizer_patient)

batch_size = 8
train_loader_patient = DataLoader(train_dataset_patient, batch_size=batch_size, shuffle=True)
val_loader_patient = DataLoader(val_dataset_patient, batch_size=batch_size, shuffle=False)
test_loader_patient = DataLoader(test_dataset_patient, batch_size=batch_size, shuffle=False)


optimizer_patient = AdamW(model_patient.parameters(), lr=2e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")
model_patient.to(device)


def train_model(model, train_loader, val_loader, optimizer, device, epochs=10):
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0
        train_loop = iter(train_loader)

        for batch in train_loop:
            optimizer.zero_grad()
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        
        model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_preds = []

        val_loop = iter(val_loader)
        with torch.no_grad():
            for batch in val_loop:
                inputs = {key: val.to(device) for key, val in batch.items()}
                labels = inputs.pop("labels")
                outputs = model(**inputs)

                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)

        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")

       
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 3:
            print(f"Early stopping after {epoch+1} epochs")
            break


train_model(model_patient, train_loader_patient, val_loader_patient, optimizer_patient, device, epochs=10)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items()}
            labels = inputs.pop("labels")
            outputs = model(**inputs)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    return accuracy, report, all_preds


test_accuracy, test_report, patient_scores = evaluate_model(model_patient, test_loader_patient, device)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Classification Report:\n{test_report}")


severity_mapping = {0: "mild", 1: "severe", 2: "critical"}
predicted_severity_scores = [severity_mapping[score] for score in patient_scores]

print("Predicted Severity Scores:")
for i, score in enumerate(predicted_severity_scores):
    print(f"Patient {i+1}: {score}")
