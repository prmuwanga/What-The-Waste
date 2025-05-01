import streamlit as st
import numpy as np
import random
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Discrete
from stable_baselines3 import PPO

#reading file for RL
df = pd.read_csv("../Datasets/Idemat_2025RevA6.csv", encoding="ISO-8859-1")

# Filter and Clean CSV
df_filtered = df[["Process", "carbon"]].copy()
df_filtered.columns = ["process", "co2_kg"]

#Strip whitespace and convert types
df_filtered["process"] = df_filtered["process"].str.strip()
df_filtered["co2_kg"] = pd.to_numeric(df_filtered["co2_kg"], errors="coerce")

# Drop bad rows
df_filtered.dropna(subset=["process", "co2_kg"], inplace=True)

# Build lookup dictionary for 
co2_lookup = dict(zip(df_filtered["process"], df_filtered["co2_kg"]))

#load agent 
agent = PPO.load("ppo_waste_disposal.zip")  

# mapping items names from dataset to categories from our CNN
cnn_class_to_process = {
    "plastic": [
        "Carbon fiber-reinforced plastic chopped - for reinforcing plastics",
        "Glassfibre - Metals open loop recycling credit dry"
    ],
    "metal": [
        "Zinc (super plastic)",
        "Lead battery cars (39 Wh per kg)", 
        "Aluminium  recycling credit closed loop (79% virgin part trade mix)", 
        "Aluminium  recycling credit closed loop (79% virgin part trade mix)", 
        "Gold  recycling credit closed loop (80% virgin part trade mix)", 
        "Magnesium  recycling credit closed loop (90% virgin part in trade mix)", 
        "Nickel  recycling credit closed loop (66% virgin part in trade mix)", 
        "Platinum  recycling credit  closed loop (88.5% virgin part in trade mix)",
        "Silver  recycling credit closed loop (45% virgin part trade mix)", 
        "Titanium  recycling credit  closed loop (81% virgin part in trade mix)", 
        "Palladium  recycling credit  closed loop (91% virgin part in trade mix)"
        
    ],
    "cardboard": [
        "Corrugated board - unbleached",
        "Corrugated board - bleached"
    ],
    "glass": [
        "Glass - packaging clear",
        "Glass - packaging mixed"
    ],
    "paper": [
        "Paper - office, recycled"
    ],
    "clothes": [
        "Cotton fibres India - rainfed",
        "Cotton fibres Bangladesh - rainfed", 
        "Cotton fibre from USA (without global seatransport)", 
        "Wool from Australia - transported to Rotterdam"
        
    ],
    "shoes": [
        "Leather Vegetable tanning - hides from Europe",
        "Leather Chrome tanning - hides from Argentina"
    ],
    "battery": [
        "AA cell battery (Alkaline)",
        "AA cell battery (Li-ion)",
        "NiCd battery AA-cell",
        "NiCd battery C-cell",
        "Lithium-ion LiCoO2 laptop battery (180 Wh/kg)",
        "Lead battery cars (39 Wh per kg)"
    ],
    "biological": [
        "Baguette white",
        "Bread multigrain",
        "Bread rye" ,
        "Crispbread",
        "White bread hard", 
        "White bread soft", 
        "Zucchini", 
        "Broccoli", 
        "Tomato", 
        "Popcorn",
        "Cake without butter"
        
    ],
    "trash": [
        "Cake with butter",
        "Residual waste - incinerated"
    ],
    "brown-glass": [
        "Beer (bottle)"
        
    ],
    "green-glass": [
        "Beer (bottle)"
    ],
    "white-glass": [
        "Beer (bottle)", 
        "Young gin",
        "Wine rose", 
        "Wine white dry"
    ]
}

#iniliaze empty dict for labels
cnn_class_to_co2 = {}

for label, materials in cnn_class_to_process.items():
    if isinstance(materials, list):
        values = [co2_lookup.get(m) for m in materials if m in co2_lookup]
        values = [v for v in values if pd.notna(v)]  # drop NaNs
        if values:
            cnn_class_to_co2[label] = sum(values) / len(values)  # average
        else:
            cnn_class_to_co2[label] = 1.0
    else:
        value = co2_lookup.get(materials, 1.0)
        cnn_class_to_co2[label] = float(value) if pd.notna(value) else 1.0



# mapping different actions for the RL 0 meaning recycle and 3 is compost
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

#creating environment for RL 

class WasteDisposalEnv(Env):
    def __init__(self, cnn_class_to_co2, correct_actions):
        super().__init__()
        self.cnn_class_to_co2 = cnn_class_to_co2
        self.correct_actions = correct_actions
        self.classes = list(cnn_class_to_co2.keys())
        self.action_space = Discrete(5)  # [Recycle, Donate, Landfill, Compost, Hazardous]
        self.observation_space = Discrete(len(self.classes))
        self.current_class = None

    # need reset, step, commonly used for gymnasium environment 
    
    def reset(self, seed=None, options=None):
        # makes a random waste selection for a new episode 
        super().reset(seed=seed)
        self.current_class = random.choice(self.classes)
        # Return its index as the observation
        obs = self.classes.index(self.current_class)
        return obs, {}

    def step(self, action):
        #get CO2 impact for the current waste class 
        co2 = self.cnn_class_to_co2[self.current_class]
        
        # correct disposal action for this waste class 
        correct_action = self.correct_actions.get(self.current_class, 2)
        
        # Special case: white-glass may be recyclable or not (adds uncertainty)
        if self.current_class == "white-glass":
            if random.random() < 0.8:
                correct_action = 0
            else:
                correct_action = 2
                
        # rewards 
        reward = co2 if action == correct_action else -co2
        
        #episode ending for each decision 
        terminated = True
        truncated = False
        
        # starts a new episode with a waste item 
        obs, _ = self.reset()
        #Return observation, reward, 
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
dropdown_options = []
for cat, materials in cnn_class_to_process.items():
    for m in (materials if isinstance(materials, list) else [materials]):
        dropdown_options.append(f"{cat}: {m}")

# Streamlit dropdown with full, specific options
selection = st.selectbox("Choose a specific material:", dropdown_options)

selected_class, selected_material = selection.split(": ", 1)


#button and loopup table options 
if st.button("Get AI Recommendation"):
    import numpy as np
    
    co2        = co2_lookup.get(selected_material, float("nan"))  # exact lookup
    obs_idx    = env.classes.index(selected_class)
    obs        = np.array([obs_idx])        # RL agent expects array shape (1,)
    action, _  = model.predict(obs, deterministic=True)
    
    st.markdown(f"Category: {selected_class}")
    st.markdown(f"Material: {selected_material}")

    if pd.isna(co2):
        st.error("CO₂ value not found for this material.")
    else:
        st.markdown(f"Estimated CO₂ impact: {co2:.2f} kg CO₂e")

    st.markdown(f"Recommended Action: {action_names[int(action)]}")
