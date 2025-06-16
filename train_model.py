import pandas as pd
import re
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import randint

# Load and label data
def load_and_label_data(true_path, fake_path):
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    
    true_df = true_df[['title', 'text']].dropna()
    fake_df = fake_df[['title', 'text']].dropna()
    
    true_df['label'] = 1  # Real = 1
    fake_df['label'] = 0  # Fake = 0

    data = pd.concat([true_df, fake_df], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    data['text'] = data['title'] + " " + data['text']
    return data[['text', 'label']]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
def preprocess_data(df):
    df['text'] = df['text'].apply(clean_text)
    return df

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main training function
def main():
    true_path = r'D:\Fake new Detection\Dataset\True.csv'
    fake_path = r'D:\Fake new Detection\Dataset\Fake.csv'

    print("📥 Loading data...")
    data = load_and_label_data(true_path, fake_path)

    print("🧹 Cleaning text...")
    data = preprocess_data(data)

    print(f"📊 Total samples: {len(data)}")
    print(f"✅ Real news samples: {sum(data['label'] == 1)}")
    print(f"❌ Fake news samples: {sum(data['label'] == 0)}")

    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    print("🔤 Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    print("🌲 Training Random Forest model...")
    param_dist = {
        'n_estimators': randint(100, 200),
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                n_iter=5, cv=2, n_jobs=-1, scoring='accuracy', verbose=1)
    search.fit(X_train_vect, y_train)
    best_model = search.best_estimator_

    print(f"🏆 Best parameters: {search.best_params_}")

    y_pred = best_model.predict(X_test_vect)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("🧾 Classification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)

    print("💾 Saving model and vectorizer...")
    os.makedirs("model", exist_ok=True)
    with open('model/fake_news_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('model/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("🎉 Model and vectorizer saved successfully in 'model/'")

if __name__ == "__main__":
    main()
