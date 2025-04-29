# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import load_img
# from sklearn.metrics import classification_report 
# import numpy as np
# import sklearn
# import os 
# import shutil 
# import random

# train_gen = ImageDataGenerator(rescale=1./255)

# # Turning every folder name into class label
# train_data = train_gen.flow_from_directory(
#     "../Datasets/combined-cleaned-dataset",
#     target_size=(224,224),
#     batch_size=32,
#     class_mode= "categorical"
# )

# print(train_data.class_indices)

# # Spliting our dataset into training, validation, and testing 
# def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#     random.seed(42)
    
#     for class_folder in os.listdir(source_dir):
#         class_path = os.path.join(source_dir, class_folder)
#         if not os.path.isdir(class_path):
#             continue 
        
#         images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
#         random.shuffle(images)
    
#         total = len(images)
#         train_end = int(total * train_ratio)
#         val_end = train_end + int(total * val_ratio)
    
#         train_images = images[:train_end]
#         val_images = images[train_end:val_end]
#         test_images = images[val_end:]
    
#         train_class_dir = os.path.join(output_dir, 'train', class_folder)
#         val_class_dir = os.path.join(output_dir, 'val', class_folder)
#         test_class_dir = os.path.join(output_dir,'test',class_folder)
    
#         os.makedirs(train_class_dir, exist_ok=True)
#         os.makedirs(val_class_dir, exist_ok=True)
#         os.makedirs(test_class_dir, exist_ok=True)
    
#         for img in train_images:
#             shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))

#         for img in val_images:
#             shutil.copy2(os.path.join(class_path, img), os.path.join(val_class_dir, img))

#         for img in test_images:
#             shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))

#         print(f" {class_folder}: {len(train_images)} train / {len(val_images)} val / {len(test_images)} test")
    
# if __name__ == "__main__":
#     source = "../Datasets/combined-cleaned-dataset"   
#     destination = "../Notebooks/the_final_sortdown"          
#     split_dataset(source, destination, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)


# # Data Augmentation
# train_gen = ImageDataGenerator(
#     rescale = 1./255,
#     rotation_range = 20,
#     width_shift_range = 0.1,
#     height_shift_range = 0.1,
#     brightness_range = [0.8, 1.2]
# )

# print("Training Data:")
# train_data = tf.keras.utils.image_dataset_from_directory(
#     "the_final_sortdown/train",
#     label_mode='categorical',
#     image_size=(224, 224),
#     batch_size=32,
#     shuffle=True
# )

# class_names = train_data.class_names

# val_gen = ImageDataGenerator(rescale=1./255)
# test_gen = ImageDataGenerator(rescale=1./255)

# print("\nValidation Data:")
# val_data = tf.keras.utils.image_dataset_from_directory(
#     "the_final_sortdown/val",
#     label_mode='categorical',
#     image_size=(224, 224),
#     batch_size=32,
#     shuffle=True
# )
# print("\nTesting Data:")
# test_data = tf.keras.utils.image_dataset_from_directory(
#     "the_final_sortdown/test",
#     label_mode='categorical',
#     image_size=(224, 224),
#     batch_size=32,
#     shuffle=False
# )



# # Transfer Learning MobileNetV2

# base_model = MobileNetV2(
#     include_top = False,
#     weights ='imagenet',
#     input_shape = (224,224,3)
# )
# base_model.trainable = False

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)

# num_classes = len(train_data.class_names)
# output = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# model.compile(
#     optimizer=Adam(),
#     loss = 'categorical_crossentropy',
#     metrics = ['accuracy']
# )

# #model.summary()

# early_stop = EarlyStopping(
#     monitor='val_loss',   # What to monitor (validation loss is typical)
#     patience=5,           # How many epochs without improvement before stopping
#     restore_best_weights=True  # Optional, but restores the best model, not the last one
# )

# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs = 7, 
#     callbacks = [early_stop]
# )

# test_loss, test_acc = model.evaluate(test_data)
# print(f"Test accuarcy: {test_acc:.2f}")

# # Adding Precision, Recall, and F1
# from sklearn.metrics import classification_report

# # Predict
# y_pred = model.predict(test_data)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Get true labels
# y_true = []
# for images, labels in test_data.unbatch():
#     y_true.append(np.argmax(labels.numpy()))
# y_true = np.array(y_true)

# # Print report
# print(classification_report(y_true, y_pred_classes, target_names=class_names))

# waste_classification_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("path_to_your_saved_model.h5")  # Change path accordingly
    return model

model = load_model()

# Define your class names
class_names = ['Plastic', 'Paper', 'Metal', 'Glass', 'Organic', 'Other']  # <- Example: replace with your real class names

# App Layout
st.title("♻️ Waste Classification App")
st.write("Upload a picture of waste and let our AI model classify it!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Important: same preprocessing as training

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
