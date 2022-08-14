import base64
from io import BytesIO

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
print("--> Starting DALL-E Server. This might take up to two minutes.")

from dalle_model import DalleModel
dalle_model = None

@app.route("/dalle", methods=["POST"])
@cross_origin()
def generate_images_api():
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_images = json_data["num_images"]
    generated_imgs = dalle_model.generate_images(text_prompt, num_images)

    returned_generated_images = []
    for idx, img in enumerate(generated_imgs):
        buffered = BytesIO()
        img.save(buffered, format="png")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {num_images} images from text prompt [{text_prompt}]")
    
    response = {'generatedImgs': returned_generated_images,
    'generatedImgsFormat': "png"}
    return jsonify(response)


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    dalle_model = DalleModel()
    dalle_model.generate_images("warm-up", 1)
    print("--> DALL-E Server is up and running!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
