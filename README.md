# SMS Spam Detection using NLP & Machine Learning

## 📌 Overview
Ever been annoyed by spam messages? This project helps **classify SMS messages as spam or ham** using **Natural Language Processing (NLP) techniques** and **Machine Learning models**. I have also compared different text vectorization methods and classification models to see which works best! 🚀

## 📂 Dataset
I am using the **Spam SMS Classification Dataset** from Kaggle. Get it here:
🔗 [Spam SMS Classification Dataset](https://www.kaggle.com/datasets/mariumfaheem666/spam-sms-classification-using-nlp)

And if you want to check out the full implementation on Kaggle:
🔗 [My Kaggle Notebook](https://www.kaggle.com/code/tapan03/spam-detection-nlp-smote-naivebayes-randomforest)

## 🛠️ Features & Techniques Used
- **Text Preprocessing**: Lowercasing, Punctuation removal, Stopwords removal, Stemming
- **Feature Extraction**: 
  - **Bag of Words (BoW)**
  - **N-Grams (Bigrams)**
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Handling Class Imbalance**: **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Model Comparisons**:
  - **Naive Bayes (GaussianNB, MultinomialNB)**
  - **Random Forest Classifier**
- **Performance Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## 📌 Installation & Usage
### 1️⃣ Clone the Repository
```bash
https://github.com/tapan2003/SMS-Spam-Detection.git
cd SMS-Spam-Detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Script
```bash
python sms_spam_detection.py
```

## 📊 Results & Comparisons
| Model            | BoW Accuracy | N-Grams Accuracy | TF-IDF Accuracy |
|-----------------|--------------|------------------|---------------|
| **GaussianNB**  | 86.7%       | 86.82%           | 86.5%        |
| **MultinomialNB** | 96.12%     | 96.64%           | 96.05%        |
| **Random Forest** | 86.62%     | 86.82%           | 97.8%        |

We can observe that **Random Forest** with **TF-IDF** vectorization performed the best.


---

🚀 **Happy Coding & No More Spam!**

