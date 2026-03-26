from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from prompt_validator import validate_prompt
from content_generator import generate_content
from evaluator import evaluate_content
import os

app = Flask(__name__, static_folder='.')
CORS(app)

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory('.', path)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    mode = data.get("mode", "text")
    topic = data.get("topic", "")
    language = data.get("language", "")

    # If the user selects Source Code, embed the requested language cleanly into the prompt context
    if mode == "code" and language:
        prompt = f"Provide {language} programming instructions and write code demonstrating: {topic}"
    else:
        prompt = topic

    
    result_dict = generate_content(prompt, topic, mode, language=language)
    content = result_dict.get("generated_content", "Error generating content.")
    context = result_dict.get("knowledge_base", "") + " " + result_dict.get("reference_content", "")
    domain_weight = result_dict.get("domain_weight", 1.0)
    
    evaluation = evaluate_content(content, context, prompt, mode, domain_weight)

    return jsonify({
        "reference_source": result_dict.get("reference_source"),
        "reference_content": result_dict.get("reference_content"),
        "external_sources": result_dict.get("external_sources"),
        "knowledge_base": result_dict.get("knowledge_base"),
        "generated_content": content,
        "evaluation": evaluation,
        "tier_counts": result_dict.get("tier_counts", {}),
        "domain_weight": domain_weight
    })

if __name__ == "__main__":
    app.run(debug=True)
