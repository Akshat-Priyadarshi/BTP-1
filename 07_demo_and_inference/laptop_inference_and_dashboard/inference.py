import torch
import numpy as np
import time
import requests
import json
import os

# =============================
# CONFIG
# =============================
PUSHBULLET_TOKEN = "<REMOVED>"

EMBEDDING_PATH = "//home/ara123/Documents/demo/data/Z_infer.npy"
LABEL_PATH     = "/home/ara123/Documents/demo/data/y_infer.npy"  
CLASSIFIER_PT  = "/home/ara123/Documents/demo/programs/classifier.pt"

SLEEP_SECONDS = 30
DEVICE = "cpu"   

#thinger.io
THINGER_USERNAME = "<REMOVED>"
THINGER_DEVICE_ID = "jetson"
THINGER_DEVICE_CRED = "<REMOVED>"

#flask app
FLASK_URL = "http://localhost:5000/update"

# =============================
# SEND STRESS LVL TO FLASK
# =============================
def send_to_flask(pred_class):
    stress_map = {
        0: " Low",
        1: " Medium",
        2: " High",
    }
    
    try:
    	response = requests.post(
    		FLASK_URL,
    		json={"stress_level": pred_class},
    		timeout=60
    	)
    	print(f"Sent to Flask: {stress_map[pred_class]} Stress -> response: {response.status_code}")
    except Exception as e:
    	print("Flask error:", e)

# =============================
# PUSHBULLET NOTIFICATION
# =============================
def send_notification(pred_class):
    stress_map = {
        0: "🟢 Low Stress",
        1: "🟡 Medium Stress",
        2: "🔴 High Stress"
    }

    title = "Stress Monitor Alert"
    body = f"Detected Stress Level: {stress_map[pred_class]}"

    data = {
        "type": "note",
        "title": title,
        "body": body
    }

    headers = {
        "Access-Token": PUSHBULLET_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        requests.post(
            "https://api.pushbullet.com/v2/pushes",
            data=json.dumps(data),
            headers=headers,
            timeout=5
        )
        print("Notification sent:", stress_map[pred_class])
    except Exception as e:
        print("Pushbullet error:", e)

# =============================
# CLASSIFIER DEFINITION
# =============================
class StressClassifier(torch.nn.Module):
    def __init__(self, embed_dim=128, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# =============================
# MAIN
# =============================
def main():
    print("Stress inference service started")

    # Load classifier
    clf = StressClassifier().to(DEVICE)
    clf.load_state_dict(torch.load(CLASSIFIER_PT, map_location=DEVICE))
    clf.eval()

    # Load embeddings
    Z = np.load(EMBEDDING_PATH)
    y = np.load(LABEL_PATH) if os.path.exists(LABEL_PATH) else None

    print(f"Loaded {len(Z)} embeddings")

    last_pred = None

    for i in range(len(Z)):
        x = torch.tensor(Z[i:i+1]).float().to(DEVICE)

        start = time.time()
        with torch.no_grad():
            pred = clf(x).argmax(dim=1).item()
        latency = (time.time() - start) * 1000

        msg = f"[{i}] Pred: {pred}"
        if y is not None:
            msg += f" | True: {y[i]}"
        msg += f" | Latency: {latency:.2f} ms"

        print(msg)

        # Send notification ONLY if stress level changes
        if pred != last_pred:
            send_to_flask(pred)
            last_pred = pred

        time.sleep(SLEEP_SECONDS)

    print("Inference completed")

# =============================
if __name__ == "__main__":
    main()

