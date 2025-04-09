
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import streamlit as st

# ------------------------------
# Config
# ------------------------------
DATASET_PATH = "C:\\Users\\mtab\\Downloads\\Ghibli\\dataset" # local dataset folder
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Dataset Loader
# ------------------------------
class GhibliDataset(Dataset):
    def __init__(self, root_dir):
        self.pairs = []
        for folder in sorted(os.listdir(root_dir)):
            pair_path = os.path.join(root_dir, folder)
            if os.path.isdir(pair_path):
                o_path = os.path.join(pair_path, "o.png")
                g_path = os.path.join(pair_path, "g.png")
                if os.path.exists(o_path) and os.path.exists(g_path):
                    self.pairs.append((o_path, g_path))

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        o_img = Image.open(self.pairs[idx][0]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        g_img = Image.open(self.pairs[idx][1]).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        return self.transform(o_img), self.transform(g_img)


# ------------------------------
# Generator Model (U-Net)
# ------------------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
           nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),      # 16x16 → 32x32
           nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),      # 32x32 → 64x64
           nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),        # 64x64 → 128x128
           nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()                               # 128x128 → 256x256
       )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


# ------------------------------
# Training Function
# ------------------------------
def train():
    print("📦 Loading dataset...")
    train_set = GhibliDataset(os.path.join(DATASET_PATH, "training"))
    val_set = GhibliDataset(os.path.join(DATASET_PATH, "validation"))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1)

    model = UNetGenerator().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for i, (o_img, g_img) in enumerate(train_loader):
            o_img, g_img = o_img.to(DEVICE), g_img.to(DEVICE)
            optimizer.zero_grad()
            output = model(o_img)
            loss = criterion(output, g_img)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "ghibli_model.pth")
    print("✅ Training complete. Model saved as `ghibli_model.pth`.")


# ------------------------------
# Inference
# ------------------------------
def generate(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor).squeeze(0).detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(output)


# ------------------------------
# Streamlit UI
# ------------------------------
def streamlit_ui():
    st.title("🎨 Ghibli Image Generator")
    uploaded = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    if uploaded:
        st.image(uploaded, caption="Original Image", use_column_width=True)

        model = UNetGenerator().to(DEVICE)
        model.load_state_dict(torch.load("ghibli_model.pth", map_location=DEVICE))
        model.eval()

        result_img = generate(model, uploaded)
        st.image(result_img, caption="Ghibli Style Output", use_column_width=True)


# ------------------------------
# Run
# ------------------------------

# Uncomment to train
#train()

# Uncomment to run UI
streamlit_ui()
