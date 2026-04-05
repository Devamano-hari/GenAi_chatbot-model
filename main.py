from prompt_validator import validate_prompt
from content_generator import generate_content
from evaluator import evaluate_content

def run():
    mode = input("Enter mode (text/code): ")
    action = input("Enter action (generate/describe/explain/give): ")
    topic = input("Enter topic: ")
    level = None
    language = None
    if mode == "text":
        level = input("Enter level (basics/advanced): ")
    elif mode == "code":
        language = input("Enter language (e.g. Python, JavaScript): ")

    prompt = validate_prompt(mode, action, topic, level, language)
    result_dict = generate_content(prompt, topic, mode)
    content = result_dict.get("generated_content", "Error generating content.")
    context = result_dict.get("knowledge_base", "") + " " + result_dict.get("reference_content", "")
    links = [src.get("url", src) if isinstance(src, dict) else src for src in result_dict.get("external_sources", [])]
    domain_weight = result_dict.get("domain_weight", 1.0)

    evaluation = evaluate_content(content, context, prompt, mode, domain_weight)

    print("\nGenerated Content:\n")
    print(content)
    print("\nEvaluation:\n", evaluation)
    print("\nVerification Links:\n", "\n".join(links))

if __name__ == "__main__":
    run()
