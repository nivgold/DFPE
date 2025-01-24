import utils
import numpy as np
from tqdm import tqdm



def process_model(dataset, model, embedding_model, batch_size=1):
    subjects = dataset.subjects

    subjects_val_predictions = {}
    subjects_test_predictions = {}
    subjects_accuracy = {}
    subjects_fingerprint = {}

    # TODO: parallel
    for subject in subjects:
        print(f"Processing subject: {subject}")

        # Process validation set
        val_predictions = []
        embeddings = []
        correct = 0
        total = 0

        val_questions = dataset.validation_subject_questions[subject]
        model_val_questions = val_questions.copy()
        num_batches = int((len(val_questions) + batch_size - 1) // batch_size)

        for batch_idx in tqdm(range(num_batches), desc=f"Validation {subject}"):
            start_idx = int(batch_idx * batch_size)
            end_idx = int(min((batch_idx + 1) * batch_size, len(model_val_questions)))
            batch_points = model_val_questions[start_idx:end_idx]
            batch_prompts = [i['prompt'] for i in batch_points]

            generated_text = model.get_model_answer(batch_prompts)

            for i, text in enumerate(generated_text):
                predicted_answer = utils.extract_answer(text)
                batch_points[i]['question_idx'] = batch_idx + i
                batch_points[i]['predicted_answer'] = predicted_answer
                batch_points[i]['generated_text'] = text

                if predicted_answer == batch_points[i]['correct_answer']:
                    correct += 1
                total += 1

                val_predictions.append(batch_points[i])

                embedding = embedding_model.encode(text)
                embeddings.append(embedding)

        # Calculate accuracy for the subject
        accuracy = correct / total if total > 0 else 0

        # Compute and save fingerprint
        embeddings = np.array(embeddings)
        fingerprint = embeddings.mean(axis=0)


        # Process test set
        # ignore test subject if not in validation
        if subject not in dataset.test_subject_questions: continue
        test_predictions = []

        test_questions = dataset.test_subject_questions[subject]
        model_test_questions = test_questions.copy()
        num_batches = int((len(model_test_questions) + batch_size - 1) // batch_size)

        for batch_idx in tqdm(range(num_batches), desc=f"Test {subject}"):
            start_idx = int(batch_idx * batch_size)
            end_idx = int(min((batch_idx + 1) * batch_size, len(model_test_questions)))
            batch_points = model_test_questions[start_idx:end_idx]
            batch_prompts = [i['prompt'] for i in batch_points]

            generated_text = model.get_model_answer(batch_prompts)

            for i, text in enumerate(generated_text):
                predicted_answer = utils.extract_answer(text)
                batch_points[i]['question_idx'] = batch_idx + i
                batch_points[i]['predicted_answer'] = predicted_answer
                batch_points[i]['generated_text'] = text

                test_predictions.append(batch_points[i])

        subjects_val_predictions[subject] = val_predictions
        subjects_test_predictions[subject] = test_predictions
        subjects_accuracy[subject] = accuracy
        subjects_fingerprint[subject] = fingerprint

    return subjects_val_predictions, subjects_test_predictions, subjects_accuracy, subjects_fingerprint