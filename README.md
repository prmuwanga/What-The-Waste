# What-The-Waste

Cindy Chen & Primah Muwanga  
Data Science 340 – Professor Gold  
April 2025

---

## Project Overview

**What The Waste** is a machine learning-powered waste classification and sustainability impact application. Our goal is to promote smarter and more sustainable recycling practices by combining computer vision and reinforcement learning to classify waste items and recommend the most environmentally responsible disposal method.

Using image classification, our model predicts the material type of waste items. A reinforcement learning agent then determines the most sustainable disposal action based on real-world carbon emission data. The app also estimates the user's carbon footprint impact and encourages better waste decisions through an interactive Streamlit interface.

---

## Key Features

- **Waste Material Classification**  
  Trained on over 18,000 images across 13 material classes using MobileNetV2.

- **Reinforcement Learning Agent**  
  A PPO (Proximal Policy Optimization) agent that optimizes disposal decisions to minimize CO₂ emissions using real carbon footprint data.

- **Environmental Impact Tracking**  
  Calculates estimated CO₂ saved or wasted per item based on the user’s disposal decision.

- **Streamlit Web Application**  
  A simple and intuitive interface where users can:
  - Upload an image
  - Receive a predicted material classification
  - Get a disposal recommendation
  - View estimated environmental impact

---

## How It Works

1. **Image Upload**  
   The user uploads or selects an image of a waste item.

2. **CNN Classification**  
   The model predicts the material class (e.g., paper, plastic, metal).

3. **Reinforcement Learning Suggestion**  
   Based on the class and CO₂ data, the PPO agent recommends the optimal disposal method (e.g., Recycle, Compost, Donate).

4. **Impact Feedback**  
   The app displays how much CO₂ is saved or lost based on the selected disposal action.

---

## Model Performance

- **Overall Accuracy:** 94%
- **High-Performing Classes:** Clothes, Paper, Biological Waste (F1-scores > 0.95)
- **Challenging Class:** Glass (F1-score = 0.33)

---

## Technologies Used

- Python
- TensorFlow / Keras (MobileNetV2)
- Stable-Baselines3 (Reinforcement Learning)
- Pandas, NumPy
- Streamlit
- Openpyxl (Excel data parsing)

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/what-the-waste.git
   cd what-the-waste
   pip install -r requirements.txt

