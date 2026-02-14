import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pennylane as qml
import joblib
import numpy as np
from PIL import Image, ImageOps

# ==========================================
# 1. SETUP & ARCHITECTURE (MUST MATCH EXACTLY)
# ==========================================
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

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    try:
        pca = joblib.load('pca_fit.pkl')
        scaler = joblib.load('scaler_fit.pkl')
        model = ParallelQuantumModel()
        model.load_state_dict(torch.load('osteo_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return resnet, pca, scaler, model, True
    except FileNotFoundError:
        return None, None, None, None, False

resnet, pca, scaler, model, loaded_success = load_resources()

# ==========================================
# 3. ADVANCED PREDICTION LOGIC (TTA)
# ==========================================
def get_prediction(img_tensor):
    """Runs a single prediction pass"""
    with torch.no_grad():
        feat_2048 = resnet(img_tensor).flatten(1).numpy()
    
    feat_6 = scaler.transform(pca.transform(feat_2048))
    
    xc = torch.tensor(feat_2048, dtype=torch.float32)
    xq = torch.tensor(feat_6, dtype=torch.float32)
    
    out = model(xc, xq)
    return torch.softmax(out, 1)[0].numpy() # Returns probabilities [p1, p2, p3]

def predict_with_tta(image):
    """
    Test-Time Augmentation:
    Predicts on Original, Flipped, and Zoomed versions, then averages.
    """
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 1. Original
    t1 = base_transform(image).unsqueeze(0)
    p1 = get_prediction(t1)
    
    # 2. Horizontal Flip
    img_flip = ImageOps.mirror(image)
    t2 = base_transform(img_flip).unsqueeze(0)
    p2 = get_prediction(t2)
    
    # 3. Slight Zoom (Center Crop)
    w, h = image.size
    img_zoom = image.crop((w*0.1, h*0.1, w*0.9, h*0.9)) # Crop 10% from edges
    img_zoom = img_zoom.resize((w, h))
    t3 = base_transform(img_zoom).unsqueeze(0)
    p3 = get_prediction(t3)
    
    # AVERAGE THE PROBABILITIES
    final_probs = (p1 + p2 + p3) / 3.0
    return final_probs

# ==========================================
# 4. APP INTERFACE
# ==========================================
st.set_page_config(page_title="Quantum Osteo AI", page_icon="🦴")

st.markdown("<h1 style='text-align: center;'>🦴 Quantum-Enhanced Osteoporosis Detection</h1>", unsafe_allow_html=True)

if not loaded_success:
    st.error("⚠️ Model files not found! Please ensure 'osteo_model.pth', 'pca_fit.pkl', and 'scaler_fit.pkl' are in the folder.")
    st.stop()

st.info("ℹ️ Uses Hybrid Quantum-Classical AI (ResNet50 + QCNN) with Test-Time Augmentation for high accuracy.")

file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original X-Ray", use_column_width=True)
    
    with col2:
        st.write("### Analysis Options")
        use_tta = st.checkbox("Enable Enhanced Accuracy (TTA)", value=True, help="Runs analysis 3 times (Original, Flipped, Zoomed) and averages results to reduce errors.")
        
        if st.button("Analyze Bone Density", type="primary"):
            with st.spinner("Processing Quantum Circuit..."):
                
                if use_tta:
                    final_probs = predict_with_tta(image)
                else:
                    # Simple single pass
                    t = transforms.Compose([
                        transforms.Resize((224, 224)), transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    final_probs = get_prediction(t(image).unsqueeze(0))

                # Results logic
                pred_idx = np.argmax(final_probs)
                result = CLASSES[pred_idx]
                confidence = final_probs[pred_idx] * 100
                
                st.markdown("---")
                if result == "Normal":
                    st.success(f"## Diagnosis: {result}")
                    st.caption("Bone density appears healthy.")
                elif result == "Osteopenia":
                    st.warning(f"## Diagnosis: {result}")
                    st.caption("Early signs of bone loss detected.")
                else:
                    st.error(f"## Diagnosis: {result}")
                    st.caption("High risk of bone fragility detected.")
                
                st.metric("AI Confidence Score", f"{confidence:.2f}%")
                
                # Bar Chart
                st.write("#### Probability Distribution")
                st.bar_chart({
                    "Normal": final_probs[0], 
                    "Osteopenia": final_probs[1], 
                    "Osteoporosis": final_probs[2]
                })
