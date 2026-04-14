# 🚀 Federated AI Medical Diagnosis System

## 🧠 Overview

This project is an **Advanced AI-powered Medical Diagnosis System** that uses:

* 🤖 Deep Learning (CNN)
* 🌐 Federated Learning (Flower Framework)
* 🔗 Blockchain-based Audit Logging
* 📊 Real-time Dashboard Visualization

The system predicts whether a **Chest X-ray is NORMAL or ABNORMAL** while maintaining **data privacy and security**.

---

## 🎯 Key Features

### 🧠 AI Model

* CNN-based classification (ABNORMAL vs NORMAL)
* Confidence score output
* Smart thresholding → ⚠️ UNCERTAIN detection

### 🛡️ X-ray Validation

* Rejects non-medical images (selfie, random images)
* Accepts only valid Chest X-rays
* Image preprocessing (resize, normalize)

### 🌐 Federated Learning

* Multiple clients (client1, client2)
* Privacy-preserving training
* Central server aggregation using Flower

### 📊 Dashboard

* Prediction history
* Confidence graph
* Accuracy vs F1 score (live)
* Confusion Matrix visualization

### 🔗 Blockchain Security

* Each prediction stored as a block
* Hash + Previous Hash linking
* Tamper detection system

### ⚡ Real-Time Monitoring

* Live training graph (Accuracy vs F1)
* API-based updates (`/metrics-data`)

---

## 🏗️ Project Structure

```
Major Project/
│
├── app.py                # Flask backend
├── client.py             # Federated client
├── server.py             # Federated server
│
├── model.pth             # Trained model
├── history.json          # Blockchain logs
├── metrics.json          # Training metrics
│
├── templates/
│   ├── index.html
│   └── history.html
│
├── dataset/
│   ├── client1/
│   └── client2/
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/project.git
cd project
```

### 2️⃣ Install Dependencies

```
pip install torch torchvision flask flwr scikit-learn opencv-python
```

---

## ▶️ How to Run

### 🔥 Step 1: Start Server

```
python server.py
```

### 🔥 Step 2: Run Clients (2 terminals)

```
python client.py
```

(Change dataset path for client1 / client2)

### 🔥 Step 3: Start Web App

```
python app.py
```

### 🌐 Open Browser

```
http://127.0.0.1:5000
```

---

## 📊 Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🔐 Security (Blockchain)

Each prediction is stored as:

```
Block = {
  data,
  timestamp,
  previous_hash,
  hash
}
```

✔ Ensures:

* Data Integrity
* Tamper Detection
* Secure Medical Logs

---

## 🧪 Technologies Used

* Python
* PyTorch
* Flask
* Flower (Federated Learning)
* OpenCV
* Chart.js
* JSON (Storage)

---

## 💡 Use Cases

* Remote Medical Diagnosis
* Privacy-Preserving AI Systems
* Hospital Data Security
* AI + Blockchain Integration

---

## 🧠 Future Enhancements

* 🔥 Tumor Heatmap (Grad-CAM)
* 📊 Multi-client comparison graph
* 🔗 Real Blockchain (Ethereum/IPFS)
* ☁️ Cloud Deployment

---

## 🎓 Viva Ready Explanation

> This project integrates Federated Learning, Blockchain, and Deep Learning to create a secure and privacy-preserving AI-based medical diagnosis system with real-time visualization.

---

## 👨‍💻 Author

**Priyanshu Rai**

---

## ⭐ Conclusion

This system demonstrates how modern AI can be combined with **privacy, security, and scalability** to build real-world healthcare solutions.

---
