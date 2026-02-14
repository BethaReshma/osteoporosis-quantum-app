import streamlit as st

# 1. UI CONFIGURATION (Must be the first command)
st.set_page_config(page_title="Quantum Osteo AI", page_icon="🦴")

# 2. SHOW LOADING STATUS IMMEDIATELY
placeholder = st.empty()
placeholder.info("⏳ System is initializing... Downloading AI models (First run takes 1-2 mins)...")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pennylane as qml
import joblib
import numpy as np
from PIL import Image

# 3. SETUP MODEL ARCHITECTURE
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

# 4. LOAD RESOURCES (Cached)
@st.cache_resource
def load_system_resources():
    # A. Load ResNet (The part that takes time to download)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    # B. Load Files
    try:
        pca = joblib.load('pca_fit.pkl')
        scaler = joblib.load('scaler_fit.pkl')
        model = ParallelQuantumModel()
        model.load_state_dict(torch.load('osteo_model.pth', map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        return None, None, None, None, str(e)
    
    return resnet, pca, scaler, model, None

# Load everything now
resnet, pca, scaler, model, error = load_system_resources()

# 5. CLEAR LOADING MESSAGE & SHOW UI
placeholder.empty() # Remove the "Initializing" message

if error:
    st.error(f"❌ Error loading files: {error}")
    st.stop()

# --- MAIN APP UI ---
st.title("🦴 Quantum-Enhanced Osteoporosis Detection")
st.markdown("### Hybrid ResNet + Quantum CNN System")

uploaded_file = st.file_uploader("Upload X-Ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Bone Density"):
        with st.spinner("🧠 Quantum Circuit is processing..."):
            # A. Transform
            t = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_t = t(image).unsqueeze(0)
            
            # B. Extract
            with torch.no_grad():
                feat_2048 = resnet(img_t).flatten(1).numpy()
            
            # C. Predict
            feat_6 = scaler.transform(pca.transform(feat_2048))
            xc = torch.tensor(feat_2048, dtype=torch.float32)
            xq = torch.tensor(feat_6, dtype=torch.float32)
            
            out = model(xc, xq)
            probs = torch.softmax(out, 1)[0]
            pred_idx = torch.argmax(probs).item()
            lbl = CLASSES[pred_idx]
            conf = float(probs[pred_idx]) * 100
            
            # D. Result
            st.markdown("---")
            if lbl == "Normal":
                st.success(f"### RESULT: Normal Bone Density")
            elif lbl == "Osteopenia":
                st.warning(f"### RESULT: Osteopenia (Early Stage)")
            else:
                st.error(f"### RESULT: Osteoporosis Detected")
                
            st.metric("Model Confidence", f"{conf:.2f}%")
            
            st.write("#### Detailed Probability:")
            cols = st.columns(3)
            cols[0].metric("Normal", f"{float(probs[0])*100:.1f}%")
            cols[1].metric("Osteopenia", f"{float(probs[1])*100:.1f}%")
            cols[2].metric("Osteoporosis", f"{float(probs[2])*100:.1f}%")
