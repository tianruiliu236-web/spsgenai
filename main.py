from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import io
import os
import random
import re

import spacy
import torch
from PIL import Image
from torchvision import transforms

from .helper_lib.model import get_model
from .helper_lib.data_loader import get_class_names

class BigramModel:
    def __init__(self, corpus):
        self.bigrams = {}
        self.build_model(corpus)

    def build_model(self, corpus):
        for sentence in corpus:
            words = re.findall(r"\b\w+\b", sentence.lower())

            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]

                if current_word not in self.bigrams:
                    self.bigrams[current_word] = []

                self.bigrams[current_word].append(next_word)

    def generate_text(self, start_word, length):
        start_word = start_word.lower()
        generated_words = [start_word]
        current_word = start_word

        for _ in range(length - 1):
            if current_word not in self.bigrams:
                break

            current_word = random.choice(self.bigrams[current_word])
            generated_words.append(current_word)

        return " ".join(generated_words)


app = FastAPI(
    title="Text Generation and Word Embedding API",
    description="A FastAPI project with bigram text generation and spaCy word embeddings."
)

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes who is falsely imprisoned and later seeks revenge.",
    "This is another example sentence.",
    "We are generating text based on bigram probabilities.",
    "Bigram models are simple but effective."
]

bigram_model = BigramModel(corpus)

nlp = spacy.load("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Text Generation and Word Embedding API"
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "start_word": request.start_word,
        "length": request.length,
        "generated_text": generated_text
    }


@app.get("/embedding")
def get_embedding(word: str):
    token = nlp.vocab[word.lower()]

    if not token.has_vector:
        raise HTTPException(
            status_code=404,
            detail=f"No embedding vector was found for '{word}'."
        )

    return {
        "word": word,
        "vector_dimension": token.vector.size,
        "embedding": token.vector.tolist()
    }
    class_names = get_class_names()

cnn_model = get_model("CNN")

model_path = "models/cifar10_cnn.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "CNN model file was not found. Run train_cnn.py first."
    )

checkpoint = torch.load(model_path, map_location="cpu")

cnn_model.load_state_dict(checkpoint["model_state_dict"])
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image."
        )

    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    return {
          "predicted_class": get_class_names()[predicted_index],
        "confidence": round(confidence, 4)
    }
 