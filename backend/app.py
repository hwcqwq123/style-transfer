from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid

from services.adam_service import run_adam
from services.lbfgs_service import run_lbfgs
from services.cyclegan_service import run_cyclegan

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_CONTENT_DIR = BASE_DIR / "uploads" / "content"
UPLOAD_STYLE_DIR = BASE_DIR / "uploads" / "style"
OUTPUT_DIR = BASE_DIR / "uploads" / "output"

for folder in [UPLOAD_CONTENT_DIR, UPLOAD_STYLE_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/api/style-transfer", methods=["POST"])
def style_transfer():
    if "content_image" not in request.files or "style_image" not in request.files:
        return jsonify({"success": False, "message": "缺少上传图片"}), 400

    content_file = request.files["content_image"]
    style_file = request.files["style_image"]
    method = request.form.get("method", "adam").lower()

    if not content_file.filename or not style_file.filename:
        return jsonify({"success": False, "message": "请选择图片"}), 400

    if not allowed_file(content_file.filename) or not allowed_file(style_file.filename):
        return jsonify({"success": False, "message": "图片格式不支持"}), 400

    task_id = uuid.uuid4().hex

    content_ext = content_file.filename.rsplit(".", 1)[1].lower()
    style_ext = style_file.filename.rsplit(".", 1)[1].lower()

    content_name = secure_filename(f"{task_id}_content.{content_ext}")
    style_name = secure_filename(f"{task_id}_style.{style_ext}")
    result_name = secure_filename(f"{task_id}_result.jpg")

    content_path = UPLOAD_CONTENT_DIR / content_name
    style_path = UPLOAD_STYLE_DIR / style_name
    output_path = OUTPUT_DIR / result_name

    content_file.save(content_path)
    style_file.save(style_path)

    try:
        if method == "adam":
            run_adam(content_path, style_path, output_path)
        elif method == "lbfgs":
            run_lbfgs(content_path, style_path, output_path)
        elif method == "cyclegan":
            run_cyclegan(content_path, style_path, output_path)
        else:
            return jsonify({"success": False, "message": "不支持的生成方法"}), 400

        return jsonify({
            "success": True,
            "message": "生成成功",
            "image_url": f"/output/{output_path.name}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"生成失败: {str(e)}"
        }), 500


@app.route("/output/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)