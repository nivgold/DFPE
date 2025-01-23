import re


def format_prompt(question, options):
    # Formats the prompt without chain-of-thought reasoning
    choices = ['A', 'B', 'C', 'D'][:len(options)]
    prompt = (
        "Answer the following multiple-choice question by providing the most appropriate response. "
        f"Answer should be one among [{', '.join(choices)}].\n\n"
    )
    prompt += f"Question: {question}\n"
    for i, option in enumerate(options):
        prompt += f"{choices[i]}. {option}\n"
    prompt += "Answer:"
    return prompt

def extract_answer(generated_text):
    # Extracts the answer letter (A-D) from the generated text
    match = re.search(r'Answer:\s*([A-D])', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        # Look for the last occurrence of A, B, C, D in the text
        letters = re.findall(r'\b([A-D])\b', generated_text.upper())
        if letters:
            return letters[-1].upper()
        else:
            return None