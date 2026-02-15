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
from PIL import Image, ImageOps # Added ImageOps for flipping

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
    # A. Load ResNet
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
placeholder.empty()

if error:
    st.error(f"❌ Error loading files: {error}")
    st.stop()

# --- HELPER FUNCTION FOR SINGLE PREDICTION ---
def get_single_prediction(img_tensor):
    """Runs one pass of the model logic"""
    # 1. ResNet Features
    with torch.no_grad():
        feat_2048 = resnet(img_tensor).flatten(1).numpy()
    
    # 2. Math Transformation (PCA + Scaler)
    feat_6 = scaler.transform(pca.transform(feat_2048))
    
    # 3. Quantum Model Prediction
    xc = torch.tensor(feat_2048, dtype=torch.float32)
    xq = torch.tensor(feat_6, dtype=torch.float32)
    
    with torch.no_grad():
        out = model(xc, xq)
        probs = torch.softmax(out, 1)[0].detach().numpy()
        
    return probs

# --- MAIN APP UI ---
st.title("🦴 Quantum-Enhanced Osteoporosis Detection")
st.markdown("### Hybrid ResNet + Quantum CNN System")

uploaded_file = st.file_uploader("Upload X-Ray Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Bone Density"):
        with st.spinner("🧠 Quantum Circuit is processing (Running Logic with TTA)..."):
            
            # --- LOGIC CHANGE START: Test-Time Augmentation (TTA) ---
            
            # Base Transform
            t = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 1. Predict on Original Image
            probs_1 = get_single_prediction(t(image).unsqueeze(0))

            # 2. Predict on Flipped Image (Mirror)
            img_flipped = ImageOps.mirror(image)
            probs_2 = get_single_prediction(t(img_flipped).unsqueeze(0))

            # 3. Predict on Zoomed Image (Center Crop 90%)
            w, h = image.size
            img_zoomed = image.crop((w*0.05, h*0.05, w*0.95, h*0.95)).resize((w, h))
            probs_3 = get_single_prediction(t(img_zoomed).unsqueeze(0))

            # 4. Average the results (Ensemble)
            final_probs = (probs_1 + probs_2 + probs_3) / 3.0
            
            # --- LOGIC CHANGE END ---

            # Extract final values for display
            pred_idx = np.argmax(final_probs)
            lbl = CLASSES[pred_idx]
            conf = float(final_probs[pred_idx]) * 100
            
            # D. Result (Exact same UI as before)
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
            # Using final_probs instead of single probs
            cols[0].metric("Normal", f"{float(final_probs[0])*100:.1f}%")
            cols[1].metric("Osteopenia", f"{float(final_probs[1])*100:.1f}%")
            cols[2].metric("Osteoporosis", f"{float(final_probs[2])*100:.1f}%")
