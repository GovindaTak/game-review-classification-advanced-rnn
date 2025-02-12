# game-review-classification-advanced-rnn
This project enhances game review sentiment classification using an Advanced RNN model with multiple hidden layers and bidirectional processing. The model predicts whether a user would recommend a game based on their review. Key steps text preprocessing (lemmatization, stopword removal, TF-IDF), feature extraction, Word2Vec embeddings, training.
-----
# 🎮 Game Review Classification using Advanced RNN  

## **Overview**  
This project builds an **Advanced Recurrent Neural Network (RNN) model** with **multiple hidden layers** to classify **game reviews** based on sentiment. The goal is to predict whether a user would **recommend** the game or not.  

This approach enhances basic RNNs by incorporating **bidirectional processing**, **word embeddings**, and **regularization techniques** for improved performance.  

---

## **Project Pipeline**  

### ✔️ **Step 1: Importing Dataset**  
- Collected **game reviews dataset**, containing:  
  - `user_review` (textual review) as input feature.  
  - `user_suggestion` (0 = Not Recommended, 1 = Recommended) as target label.  

### ✔️ **Step 2: Text Preprocessing**  
- Applied **lemmatization, stopword removal, punctuation cleanup** using **spaCy**.  
- Performed **POS Tagging & Named Entity Recognition (NER)** for better text representation.  

### ✔️ **Step 3: Feature Engineering (Embeddings)**  
- Implemented **TF-IDF Vectorization** for baseline feature extraction.  
- Built **100-dimensional Word2Vec embeddings** trained on the dataset.  

### ✔️ **Step 4: Advanced RNN Model Implementation**  
- **Bi-Directional Multi-Layer RNN** using **PyTorch**.  
- Applied **Leaky ReLU activation** and **Dropout Regularization**.  
- Trained for **30 epochs** with **early stopping**.  

---

## **Dataset Details**  
- **Total Records:**  
  - **Training Set:** 17,877  
  - **Validation & Testing Set:** 1,000+  
- **Feature:** `user_review` (text data)  
- **Target Variable:** `user_suggestion` (0 = Not Recommended, 1 = Recommended)  

---

## **Advanced RNN Model Architecture**  
- **Input Layer:** Pre-trained **100-dimensional Word2Vec embeddings**  
- **Hidden Layers:** Multi-layer **Bi-Directional RNN**  
- **Activation Function:** **Leaky ReLU**  
- **Regularization:** Dropout (Rate = `0.5`)  
- **Optimizer:** Adam (`lr=0.001`)  
- **Loss Function:** Binary Cross-Entropy Loss  

---

## **Results & Accuracy**  
🏆 **Best Model Performance:**  
✅ **Training Accuracy:** `56.72%`  
✅ **Validation Accuracy:** `56.87%`  
✅ **Lowest Validation Loss:** `0.6839` at **Epoch 7**  

📊 **Next Steps:**  
- Implement **LSTMs and GRUs** for enhanced performance.  
- Fine-tune **hyperparameters and embeddings** for better results.  
- Explore **Transformer-based models (BERT, GPT-3, T5)**.  

---

## **Installation & Usage**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/GovindaTak/game-review-classification-advanced-rnn.git
cd game-review-classification-advanced-rnn
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Model**  
```bash
python train_model.py
```

---

## **Technologies Used**  
🛠 **Libraries & Tools:**  
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Natural Language Processing (NLP)**: spaCy, NLTK, TF-IDF, Word2Vec  
- **Machine Learning**: scikit-learn  
- **Deep Learning**: PyTorch  
- **Data Visualization**: Matplotlib, Seaborn  

---

## **Contributing**  
Want to improve the model? Feel free to **fork the repo**, create a new branch, and submit a **pull request**. 🚀  

---

## **Connect with Me**  
📌 **Email** : govindatak19@gmail.com

💡 **Let's build better AI models together!** 🚀🔥  
