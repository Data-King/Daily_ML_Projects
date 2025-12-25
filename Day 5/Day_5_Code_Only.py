"""
Fake Review Detection System
Uses NLP + Transformer (BERT) for detecting bot-generated reviews
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

class FakeReviewDataset(Dataset):
    """Custom Dataset for fake review detection"""
    
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class FakeReviewDetectionSystem:
    """
    A fake review detection system using BERT transformer model
    for identifying bot-generated reviews
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
    def create_synthetic_data(self, n_samples=1000):
        """
        Create synthetic fake review dataset for demonstration
        In practice, you would load real labeled data
        """
        # Fake reviews (bot-generated patterns)
        fake_reviews = [
            "Amazing product! Best purchase ever! 5 stars! Highly recommend!",
            "This is the best thing I ever bought. Perfect in every way. Must buy!",
            "Excellent quality! Fast shipping! Great seller! Will buy again!",
            "Outstanding product! Exceeded expectations! Perfect! Recommended!",
            "Best product ever! Amazing quality! Super fast delivery! 5 stars!",
            "Fantastic! Great value! Perfect condition! Highly satisfied!",
            "Wonderful product! Excellent service! Fast shipping! Recommended!",
            "Perfect! Amazing! Best ever! Great quality! Will purchase again!",
            "Super happy with purchase! Excellent product! Fast delivery!",
            "Best decision ever! Perfect quality! Great price! Highly recommend!"
        ]
        
        # Real reviews (more nuanced, specific)
        real_reviews = [
            "The product works as described, though the battery life could be better. Decent for the price.",
            "I like the design and it's comfortable to use. Took a while to arrive but quality is good.",
            "Pretty good overall. The color is slightly different from the photos but I'm satisfied.",
            "Works well for my needs. Assembly instructions could be clearer. Material feels durable.",
            "Decent product. Does what it's supposed to do. Shipping was slower than expected.",
            "Good value for money. The size is perfect for my space. Minor scratches on arrival.",
            "I've been using it for two weeks. So far so good, though it's a bit noisy.",
            "Meets my expectations. The quality is acceptable for this price range. Quick delivery.",
            "Not bad. The interface takes some getting used to but it works reliably.",
            "Pretty satisfied with the purchase. Would have preferred more color options."
        ]
        
        # Generate dataset
        reviews = []
        labels = []
        
        for _ in range(n_samples // 2):
            reviews.append(np.random.choice(fake_reviews))
            labels.append(1)  # 1 = fake
            
            reviews.append(np.random.choice(real_reviews))
            labels.append(0)  # 0 = real
        
        df = pd.DataFrame({
            'review': reviews,
            'label': labels
        })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def prepare_data(self, df, test_size=0.2, val_size=0.1, batch_size=16):
        """Prepare data loaders for training"""
        # Split data
        train_val_data, test_data = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        train_data, val_data = train_test_split(
            train_val_data, test_size=val_size/(1-test_size), 
            random_state=42, stratify=train_val_data['label']
        )
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        train_dataset = FakeReviewDataset(
            train_data['review'].values,
            train_data['label'].values,
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = FakeReviewDataset(
            val_data['review'].values,
            val_data['label'].values,
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = FakeReviewDataset(
            test_data['review'].values,
            test_data['label'].values,
            self.tokenizer,
            self.max_length
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_data, val_data, test_data
    
    def initialize_model(self):
        """Initialize BERT model for classification"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
        
        self.model.to(self.device)
        
    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy, predictions, true_labels
    
    def train_model(self, epochs=3, learning_rate=2e-5):
        """Train the model"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(optimizer, scheduler)
            
            # Validate
            val_loss, val_accuracy, _, _ = self.evaluate(self.val_loader)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'best_fake_review_detector.pth')
    
    def evaluate_model(self):
        """Evaluate model on test set with detailed metrics"""
        test_loss, test_accuracy, predictions, true_labels = self.evaluate(self.test_loader)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        return test_accuracy, predictions, true_labels
    
    def predict_review(self, review_text):
        """
        Predict whether a single review is fake or real
        
        Parameters:
        -----------
        review_text : str
            The review text to analyze
        """
        self.model.eval()
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    
    def analyze_multiple_reviews(self, reviews):
        """Analyze multiple reviews at once"""
        results = []
        for i, review in enumerate(reviews, 1):
            pred, conf = self.predict_review(review)
            results.append({
                'review': review,
                'prediction': 'Fake' if pred == 1 else 'Real',
                'confidence': conf
            })
        
        return results


# Main execution
if __name__ == "__main__":
    # Initialize system
    system = FakeReviewDetectionSystem(max_length=128)
    
    # Step 1: Create/Load data
    df = system.create_synthetic_data(n_samples=1000)
    
    # Step 2: Prepare data
    train_data, val_data, test_data = system.prepare_data(df, batch_size=16)
    
    # Step 3: Initialize model
    system.initialize_model()
    
    # Step 4: Train model
    system.train_model(epochs=3, learning_rate=2e-5)
    
    # Step 5: Evaluate model
    system.evaluate_model()
    
    # Step 6: Test individual predictions
  
    test_reviews = [
        "Amazing product! Best purchase ever! 5 stars! Highly recommend!",
        "The product works well but the battery could last longer. Good value overall.",
        "Perfect! Excellent! Outstanding! Best ever! Must buy now!",
        "I've been using this for a month. It does the job, though setup was tricky."
    ]
    
    system.analyze_multiple_reviews(test_reviews)

