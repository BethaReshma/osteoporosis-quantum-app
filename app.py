import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pennylane as qml
import joblib
import numpy as np
from PIL import Image

# 1. SETUP & MODEL ARCHITECTURE
N_QUBITS = 6
CLASSES = ["Normal", "Osteopenia", "Osteoporosis"]

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def q_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class ParallelQuantumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_layer = nn.Sequential(nn.Linear(2048, 64), nn.ReLU())
        self.q_layer = qml.qnn.TorchLayer(q_circuit, {"weights": (3, N_QUBITS, 3)})
        self.q_dense = nn.Linear(N_QUBITS, 16)
        self.final = nn.Sequential(nn.Linear(64+16, 3))
    def forward(self, xc, xq):
        c = self.c_layer(xc)
        q = self.q_dense(self.q_layer(xq))
        return self.final(torch.cat((c, q), dim=1))

# 2. LOAD EVERYTHING
@st.cache_resource
def load_resources():
    # Load ResNet
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    # Load Math Tools
    pca = joblib.load('pca_fit.pkl')
    scaler = joblib.load('scaler_fit.pkl')
    
    # Load Quantum Model
    model = ParallelQuantumModel()
    model.load_state_dict(torch.load('osteo_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    return resnet, pca, scaler, model

resnet, pca, scaler, model = load_resources()

# 3. APP INTERFACE
st.title("🦴 Quantum Osteoporosis Detection")
st.write("Upload an X-ray to analyze bone density.")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)
    
    if st.button("Analyze"):
        with st.spinner("Processing..."):
            # A. Prepare Image
            t = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_t = t(image).unsqueeze(0)
            
            # B. Extract Features
            with torch.no_grad():
                feat_2048 = resnet(img_t).flatten(1).numpy()
                
            # C. Apply Saved Math Rules
            feat_6 = scaler.transform(pca.transform(feat_2048))
            
            # D. Predict
            xc = torch.tensor(feat_2048, dtype=torch.float32)
            xq = torch.tensor(feat_6, dtype=torch.float32)
            
            out = model(xc, xq)
            probs = torch.softmax(out, 1)[0]
            pred = torch.argmax(probs).item()
            
            # E. Show Result
            result = CLASSES[pred]
            color = "green" if result == "Normal" else "orange" if result == "Osteopenia" else "red"
            st.markdown(f"## Result: :{color}[{result}]")
            st.progress(float(probs[pred]))
            st.write(f"Confidence: {float(probs[pred])*100:.2f}%")