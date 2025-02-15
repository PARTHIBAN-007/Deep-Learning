# Deep Learning

## Overview
Deep Learning is a subfield of Machine Learning that involves the use of neural networks with many layers to learn complex and hidden patterns in the data. Neural Networks are inspired by the biological structure and function of the human brain. It is an effective way to process large amounts of data and make predictions. Deep learning is computationally intensive as it requires large amounts of data, and the performance of a deep learning model is directly proportional to the amount of data available.

<img src="https://lh7-us.googleusercontent.com/lD2mGUzyMe9YtPkipb_sVFSy3H0tYtdCf1dEipeXAf3o60rHrpxj3OJblK5hH1tSNmkxDd9fd-z3RvkbX021SPxEVC67NT4AVbIkOM76G6aVv_2b7StWIbikPWy8qJmYjlThxWKODvp4afmpH2A3p4w" style="height:400px; width:700px; " >

## Machine Learning vs Deep Learning
- **Machine Learning:** Preferable when the data is low and interpretability is key. It requires manual feature extraction.
- **Deep Learning:** Shines with large-scale data and complex tasks like image recognition and voice recognition, though it often requires computational power and vast amounts of data.

![ML vs DL](https://www.softwaretestinghelp.com/wp-content/qa/uploads/2019/04/DeepLearning.png)

---

# Neural Network Architecture
## Main Components of a Neural Network:
- **Input Layer**
- **Hidden Layer(s)**
- **Output Layer**
- **Neurons**
- **Weights and Biases**
- **Activation Functions**
- **Loss Functions**
- **Optimizers**

### Explanation:
- **Input Layer:** Receives raw input data (features) where each node represents a feature.
- **Hidden Layer:** Intermediate layers between input and output that transform data into higher-level representations, allowing the network to learn complex patterns.
- **Output Layer:** Produces the final prediction or classification, with the number of neurons representing the output classes.
- **Neurons:** Computational units that receive input, apply a weighted sum and an activation function, and produce an output.
- **Weights and Biases:** Parameters that adjust during training to minimize the error between actual and predicted values.
- **Activation Functions:** Non-linear functions applied to neurons to enable learning of complex patterns.
- **Loss Functions:** Measure the error in predictions.
- **Optimizers:** Algorithms that update weights and biases to minimize the loss function.

### Neural Network Diagram:
<div style="display:grid; grid-template-columns:repeat(2,1fr); gap:10px;">
    <img src="https://som.edu.vn/wp-content/uploads/2023/12/deep-neural-networks-la-gi.png" style="height:400px; width:500px;">
    <img src="https://miro.medium.com/v2/resize:fit:1200/1*qQPpdtR0r1APiEfTqN74aA.png" style="height:400px; width:500px;>
</div>

---

# Deep Learning Architectures

## Common Deep Learning Models:
- **Artificial Neural Network (ANN):** The foundational deep learning model consisting of interconnected neurons. Used for classification and regression tasks.
- **Convolutional Neural Network (CNN):** Primarily used for image and spatial data processing, utilizing convolutional layers to automatically capture features like edges and textures.
- **Recurrent Neural Network (RNN):** Designed for sequential data (e.g., time series, text) and maintains information across time steps.
- **Long Short-Term Memory (LSTM):** A type of RNN that uses memory cells and gating mechanisms (input, output, forget gates) to capture long-term dependencies and prevent vanishing gradients.
- **Gated Recurrent Unit (GRU):** A simplified variant of LSTM with fewer gates, making it computationally more efficient while still handling long-range dependencies effectively.

---

# Applications of Deep Learning
- **Image Recognition:** Used in facial recognition, object detection, and medical imaging (e.g., tumor detection in MRIs).
- **Natural Language Processing (NLP):** Powers applications like language translation, sentiment analysis, chatbots, and text generation (e.g., BERT, GPT models).
- **Speech Recognition:** Enables voice assistants (e.g., Siri, Alexa) to convert spoken language into text and respond accordingly.
- **Autonomous Vehicles:** Used in perception systems for detecting objects, understanding the environment, and making driving decisions.
- **Recommender Systems:** Suggests products, movies, or music based on user preferences (e.g., Netflix, Amazon).
- **Financial Forecasting:** Predicts stock prices, credit scoring, fraud detection, and risk assessment.
- **Healthcare:** Applications in disease diagnosis, drug discovery, personalized treatment recommendations, and medical data analysis.
- **Gaming:** Neural networks enhance game AI by learning to make strategic decisions dynamically.

---
