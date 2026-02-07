import torch
from PIL import Image
import torchvision.transforms as T
import argparse

# -------------------
# 1. Define your model
# -------------------
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(3,64,3,1,1), torch.nn.ReLU())
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(64,128,3,2,1), torch.nn.ReLU())
        self.dec1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(128,64,4,2,1), torch.nn.ReLU())
        self.out  = torch.nn.Conv2d(64,3,3,1,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = self.dec1(e2)
        return torch.sigmoid(self.out(d1))

# -------------------
# 2. Load model
# -------------------
def load_model():
    model = UNet()
    model.load_state_dict(torch.load("lol_enhancer.pth", map_location="cpu"))
    model.eval()
    return model

# -------------------
# 3. Enhance image
# -------------------
def enhance(input_path, output_path):
    model = load_model()
    transform = T.Compose([
        T.Resize((256,256)),
        T.ToTensor()
    ])

    img = Image.open(input_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        y = model(x).squeeze(0).permute(1,2,0).numpy()

    y = (y * 255).clip(0,255).astype("uint8")
    out = Image.fromarray(y)
    out.save(output_path)
    print("Saved:", output_path)

# -------------------
# CLI
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    enhance(args.input, args.output)
