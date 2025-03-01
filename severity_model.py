import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any

class SeverityModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "dmis-lab/biobert-v1.1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        ).to(self.device)
        self.severity_mapping = {0: "mild", 1: "severe", 2: "critical"}

    def predict_severity(self, text: str) -> Dict[str, Any]:
        # Prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

        # Convert to severity score (1-10 scale)
        severity_score = (prediction + 1) * 3.33  # Maps 0,1,2 to roughly 3,6,9

        return {
            'score': round(severity_score),
            'confidence': confidence,
            'classification': self.severity_mapping[prediction]
        }

    def train(self, texts, labels, epochs=5):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(epochs):
            total_loss = 0
            for text, label in zip(texts, labels):
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                labels_tensor = torch.tensor([label]).to(self.device)
                
                outputs = self.model(**inputs, labels=labels_tensor)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Average loss: {total_loss/len(texts):.4f}")

# Initialize model
severity_model = SeverityModel()
