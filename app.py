import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pennylane as qml
import joblib
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# 1. UI CONFIGURATION
st.set_page_config(page_title="Quantum Osteo AI", page_icon="🦴")

# 2. SHOW LOADING STATUS
placeholder = st.empty()
placeholder.info("⏳ System is initializing... Downloading AI models...")

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

# 4. LOAD RESOURCES
@st.cache_resource
def load_system_resources():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    try:
        pca = joblib.load('pca_fit.pkl')
        scaler = joblib.load('scaler_fit.pkl')
        model = ParallelQuantumModel()
        model.load_state_dict(torch.load('osteo_model.pth', map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        return None, None, None, None, str(e)
    
    return resnet, pca, scaler, model, None

resnet, pca, scaler, model, error = load_system_resources()

placeholder.empty()

if error:
    st.error(f"❌ Error loading files: {error}")
    st.stop()

# --- PREDICTION LOGIC ---
def get_raw_probs(img_tensor):
    with torch.no_grad():
        feat_2048 = resnet(img_tensor).flatten(1).numpy()
    feat_6 = scaler.transform(pca.transform(feat_2048))
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
        with st.spinner("🧠 Quantum Circuit is processing..."):
            
            # --- STRATEGY: CONTRAST BOOST + SENSITIVITY BIAS ---
            
            t = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 1. Prediction on Normal Image
            p1 = get_raw_probs(t(image).unsqueeze(0))

            # 2. Prediction on High Contrast Image (Helps see bone porosity)
            enhancer = ImageEnhance.Contrast(image)
            img_contrast = enhancer.enhance(1.5) # Increase contrast by 50%
            p2 = get_raw_probs(t(img_contrast).unsqueeze(0))

            # 3. Prediction on Zoomed Crop (Focus on texture)
            w, h = image.size
            img_zoom = image.crop((w*0.15, h*0.15, w*0.85, h*0.85)).resize((w, h))
            p3 = get_raw_probs(t(img_zoom).unsqueeze(0))

            # 4. Average Ensemble
            # Weights: Normal(30%), Contrast(30%), Zoom(40%)
            avg_probs = (p1 * 0.3) + (p2 * 0.3) + (p3 * 0.4)

            # 5. MEDICAL CALIBRATION (CRITICAL STEP)
            # Problem: Model favors "Normal". 
            # Solution: Apply a bias to boost Sensitivity for Osteoporosis.
            # We multiply Osteoporosis score by 1.4 (40% boost) and Normal by 0.8 (20% penalty)
            
            calibrated_probs = np.array(avg_probs) # Copy
            calibrated_probs[2] = avg_probs[2] * 1.4  # Boost Osteoporosis
            calibrated_probs[1] = avg_probs[1] * 1.1  # Slight Boost Osteopenia
            calibrated_probs[0] = avg_probs[0] * 0.85 # Penalize Normal
            
            # Re-normalize to sum to 100%
            final_probs = calibrated_probs / calibrated_probs.sum()

            # --- END LOGIC ---

            # Results Display
            pred_idx = np.argmax(final_probs)
            lbl = CLASSES[pred_idx]
            conf = float(final_probs[pred_idx]) * 100
            
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
            cols[0].metric("Normal", f"{float(final_probs[0])*100:.1f}%")
            cols[1].metric("Osteopenia", f"{float(final_probs[1])*100:.1f}%")
            cols[2].metric("Osteoporosis", f"{float(final_probs[2])*100:.1f}%")
