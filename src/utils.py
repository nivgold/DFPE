import re
import numpy as np
from collections import defaultdict


def format_prompt(question, choices):
    # Formats the prompt without chain-of-thought reasoning
    choices = ['A', 'B', 'C', 'D'][:len(choices)]
    prompt = (
        "Answer the following multiple-choice question by providing the most appropriate response. "
        f"Answer should be one among [{', '.join(choices)}].\n\n"
    )
    prompt += f"Question: {question}\n"
    for i, option in enumerate(choices):
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


def calculate_best_single_metrics(models_subjects_predictions):
    subjects_models_accuracies = defaultdict(dict)
    models_overall_accuracies = {}

    for model in models_subjects_predictions:
        model_correct = 0
        model_total = 0
        for subject in models_subjects_predictions[model]:
            model_subject_correct = 0
            model_subject_total = 0
            for question in models_subjects_predictions[model][subject]:
                if question['predicted_answer'] == question['correct_answer']:
                    model_correct += 1
                    model_subject_correct += 1
                model_subject_total += 1
                model_total += 1
            subjects_models_accuracies[subject][model] = model_subject_correct / model_subject_total
        models_overall_accuracies[model] = model_correct / model_total

    # calculate overall-accuracy best
    models_overall_accuracies['best'] = max(list(models_overall_accuracies.values()))

    # calculate subject-level accuracy best
    for subject in subjects_models_accuracies:
        subjects_models_accuracies[subject]['best'] = np.max(list(subjects_models_accuracies[subject].values()))

    subjects_models_accuracies['average']['average'] = np.average([i['best'] for i in list(subjects_models_accuracies.values())])

    return models_overall_accuracies, subjects_models_accuracies