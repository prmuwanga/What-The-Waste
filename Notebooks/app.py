import streamlit as st
import numpy as np
import random
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO

df = pd.read_csv("../Datasets/Idemat_2025RevA6.csv", encoding="ISO-8859-1")

# Select only the needed columns
df_filtered = df[[ "Process", "carbon"]]

# Rename columns for easier access
df_filtered.columns = ["process", "co2_kg"]
df_filtered.dropna(subset=["process", "co2_kg"], inplace=True)

co2_lookup = dict(zip(df_filtered["process"], df_filtered["co2_kg"]))
print(df_filtered.head())

agent = PPO.load("ppo_waste_disposal.zip")  

cnn_class_to_process = {
    "plastic": "Polyethylene (Europe)",  
    "metal": "Aluminium sheet rolling Europe",  
    "cardboard": "Corrugated board - unbleached",  
    "glass": "Glass - packaging mixed", 
    "paper": "Newsprint paper - 100% recycled",  
    "clothes": "Cotton fibres India - rainfed", 
    "shoes": "Leather Vegetable tanning - hides from Europe",  
    "battery": "AA cell battery (Alkaline)",  
    "biological": "Food waste - average",  
    "trash": "Municipal solid waste - unsorted",  
    "brown-glass": "Glass - packaging brown", 
    "green-glass": "Glass - packaging green",  
    "white-glass": "Glass - packaging clear" 
}


cnn_class_to_co2 = {
    label: float(co2_lookup.get(material, 1.0))
    for label, material in cnn_class_to_process.items()
}


correct_actions = {
    "cardboard": 0,   # Recycle
    "clothes": 1,     # Donate
    "battery": 4,     # Hazardous
    "brown-glass":0,  # Recycle
    "glass": 0,       # Recycle
    "green-glass" : 0,# Recycle
    "metal": 0,       # Recycle
    "paper": 0,       # Recycle
    "plastic": 0,     # Recycle
    "shoes": 1,       # Donate
    "white-glass": 0, # Recycle 
    "trash" : 2,      # Landfill
    "biological": 3   # Compost
    
}

action_names = {0:"Recycle",1:"Donate",2:"Landfill",3:"Compost",4:"Hazardous"}

class WasteDisposalEnv(Env):
    def __init__(self, cnn_class_to_co2, correct_actions):
        super().__init__()
        self.cnn_class_to_co2 = cnn_class_to_co2
        self.correct_actions = correct_actions
        self.classes = list(cnn_class_to_co2.keys())
        self.action_space = Discrete(5)
        self.observation_space = Discrete(len(self.classes))
        self.current_class = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_class = random.choice(self.classes)
        obs = self.classes.index(self.current_class)
        return obs, {}

    def step(self, action):
        co2 = self.cnn_class_to_co2[self.current_class]
        correct_action = self.correct_actions.get(self.current_class, 2)
        if self.current_class == "white-glass":
            if random.random() < 0.8:
                correct_action = 0
            else:
                correct_action = 2
        reward = co2 if action == correct_action else -co2
        terminated = True
        truncated = False
        obs, _ = self.reset()
        return obs, reward, terminated, truncated, {
            "class": self.current_class,
            "co2": co2,
            "action": action,
            "correct": correct_action
        }
        
env = WasteDisposalEnv(cnn_class_to_co2, correct_actions)

st.title("♻️ Waste Disposal RL Demo")
st.write("Pick a waste type and let the agent decide how to dispose it.")

# Load agent
model = PPO.load("ppo_waste_disposal")
env = WasteDisposalEnv(cnn_class_to_co2, correct_actions)

# User input
choice = st.selectbox("Choose a waste item:", env.classes)

if st.button("Get AI Recommendation"):
    import numpy as np
    obs = env.classes.index(choice)
    obs = np.array([obs])  

    action, _ = model.predict(obs, deterministic=True)
    co2 = cnn_class_to_co2[choice]
    
    st.markdown(f" Recommended Action: `{action_names[int(action)]}`")

    if pd.isna(co2):
        st.error("⚠️ CO₂ value not found or is invalid for this material.")
    else:
        st.markdown(f" Estimated CO₂ impact: `{co2:.2f} kg CO₂e`")
    
    st.write(f"Raw CO₂ value: {co2} (type: {type(co2)})")


    
    