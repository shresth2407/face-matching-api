from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import insightface
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image

# Initialize Flask
app = Flask(__name__)

# Load ArcFace model
print("🔄 Loading ArcFace model (buffalo_s)...")
face_app = FaceAnalysis(name='buffalo_s')
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Model loaded successfully!")

def read_image(file_data):
    """Read image from base64 or file"""
    if isinstance(file_data, str):
        # base64 image
        image_data = base64.b64decode(file_data.split(',')[-1])
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # direct file upload
        return cv2.imdecode(np.frombuffer(file_data.read(), np.uint8), cv2.IMREAD_COLOR)

@app.route('/')
def home():
    return jsonify({"message": "Face Matching API using ArcFace (buffalo_s)"}), 200

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        # Accept either base64 or uploaded files
        if 'image1' not in request.files and 'image1' not in request.json:
            return jsonify({"error": "image1 and image2 required"}), 400

        # Read images
        if request.files:
            img1 = read_image(request.files['image1'])
            img2 = read_image(request.files['image2'])
        else:
            img1 = read_image(request.json['image1'])
            img2 = read_image(request.json['image2'])

        # Get faces and embeddings
        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)

        if len(faces1) == 0 or len(faces2) == 0:
            return jsonify({"match": False, "message": "No face detected in one or both images"}), 400

        # Use the first face found
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding

        # Compute cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Threshold tuning (ArcFace typically ~0.3-0.4 for same identity)
        is_match = sim > 0.35

        return jsonify({
            "match": bool(is_match),
            "similarity": float(sim),
            "message": "Faces Match ✅" if is_match else "Faces Do Not Match ❌"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
