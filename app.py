from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import pickle
import gdown
import logging

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Drive file IDs (replace with actual IDs)
MODEL_FILE_ID = "1b0nQXA6JGARbrnznc00DgsmM0hwri9-R"
FEATURES_FILE_ID = "1eqjceJ4sXmWnl2FZNHUPZRBVd-ShHNhK"
TOKENIZER_FILE_ID = "1vjfd6URHSH4OYGzTmNM21aUbwP4-2Gfq"

# Function to download files if they don't exist
def download_file(file_id, output):
    if not os.path.exists(output):
        print(f"Downloading {output} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# Download required files
download_file(MODEL_FILE_ID, "best_model.h5")
download_file(FEATURES_FILE_ID, "features.pkl")
download_file(TOKENIZER_FILE_ID, "tokenizer.pkl")

# Load the trained model for captioning and VGG16 for feature extraction
try:
    model = load_model("best_model.h5")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

vgg_model = VGG16(weights="imagenet")
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Load the tokenizer
try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    logging.error("Tokenizer file not found. Make sure 'tokenizer.pkl' is in the same directory.")
    tokenizer = None

# Define max_length for captions
MAX_LENGTH = 35  # Adjust this based on your trained model's max sequence length

def idx_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None) if tokenizer else None

def word_to_index(word, tokenizer):
    return tokenizer.word_index.get(word, 0) if tokenizer else 0  # Default to 0 if word not found

def predict_caption(model, feature):
    if model is None:
        return "Error: Model not loaded."

    in_text = "startseq"
    for _ in range(MAX_LENGTH):
        sequence = [word_to_index(word, tokenizer) for word in in_text.split() if word_to_index(word, tokenizer) > 0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH)

        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None or word == "endseq":
            break
        in_text += " " + word

    return in_text.replace("startseq ", "").strip()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/tool", methods=["GET", "POST"])
def tool():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            try:
                feature = vgg_model.predict(image, verbose=0)
                predicted_caption = predict_caption(model, feature)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                predicted_caption = "Error generating caption"

            return render_template(
                "tool.html", 
                uploaded_image=filepath,
                predicted_caption=predicted_caption
            )
    return render_template("tool.html")

if __name__ == "__main__":
    app.run(debug=True)
