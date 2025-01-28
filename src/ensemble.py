from sklearn.cluster import DBSCAN
import numpy as np


def cluster_models_per_subject(fingerprints, subjects, esp=3):
    model_names = fingerprints.keys()
    cluster_labels = {}

    for subject in subjects:
        subject_fingerprints = []
        model_list = []
        for model_name in model_names:
            if subject in fingerprints[model_name]:
                fingerprint = fingerprints[model_name][subject]
                subject_fingerprints.append(fingerprint)
                model_list.append(model_name)
        if len(subject_fingerprints) >= esp:
            subject_fingerprints = np.stack(subject_fingerprints)
            clustering_model = DBSCAN(min_samples=1, eps=esp, metric="cosine")
            labels = clustering_model.fit_predict(subject_fingerprints)
            cluster_labels[subject] = {model_name: label for model_name, label in zip(model_list, labels)}
        else:
            # Not enough models for clustering; assign all to one cluster
            cluster_labels[subject] = {model_name: 0 for model_name in model_list}
    return cluster_labels

def calculate_mean_accuracy_per_subject(model_accuracies, subjects):
    model_names = model_accuracies.keys()

    # Compute mean accuracy per subject across all models
    mean_accuracies = {}
    for subject in subjects:
        accuracies = []
        for model_name in model_names:
            acc = model_accuracies.get(model_name, {}).get(subject, None)
            if acc is not None:
                accuracies.append(acc)
        if accuracies:
            mean_accuracies[subject] = float(np.mean(accuracies))
        else:
            mean_accuracies[subject] = 0.0  # If no accuracies, set to 0
    return mean_accuracies

def select_representative_models(subject, cluster_labels, model_accuracies):
    selected_models = []
    clusters = set(cluster_labels[subject].values())
    for cluster_label in clusters:
        # Models in this cluster
        models_in_cluster = [model_name for model_name, label in cluster_labels[subject].items() if label == cluster_label]
        # Select the model with the highest accuracy
        best_accuracy = -1
        best_model = None
        for model_name in models_in_cluster:
            accuracy = model_accuracies[model_name].get(subject, 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        if best_model:
            selected_models.append(best_model)
    return selected_models

def ensemble_predictions_on_test_set(selected_models, models_subject_test_predictions, models_subject_accuracies, quantile_param, scaling_factor=5.0):
    # Get accuracies for weighting
    accuracies_array = np.array([models_subject_accuracies[model] for model in selected_models])

    # Compute the quantile threshold
    threshold = np.quantile(accuracies_array, quantile_param)
    above_threshold = accuracies_array >= threshold

    if not np.any(above_threshold):
        # If no models above threshold, use all models with equal weights
        weights = np.ones_like(accuracies_array) / len(accuracies_array)
    else:
        # Set weights to zero for models below threshold
        adjusted_accuracies = np.where(above_threshold, accuracies_array, 0)
        # Apply exponential scaling to accuracies above threshold
        scaled_accuracies = np.exp(adjusted_accuracies * scaling_factor)
        total_scaled_accuracy = np.sum(scaled_accuracies)
        if total_scaled_accuracy > 0:
            weights = scaled_accuracies / total_scaled_accuracy
        else:
            # If total is zero, assign equal weights to models above threshold
            num_models_above_threshold = np.sum(above_threshold)
            weights = np.where(above_threshold, 1.0 / num_models_above_threshold, 0.0)

    # Initialize metrics
    mvoting_correct = 0
    dfpe_correct = 0
    total = 0

    test_questions = list(models_subject_test_predictions.values())[0]
    num_questions = len(test_questions)

    for idx in range(num_questions):
        correct_answer = test_questions[idx]['correct_answer']
        choices = test_questions[idx]['choices']
        choice_labels = ['A', 'B', 'C', 'D'][:len(choices)]
        choice_probs = {choice: 0.0 for choice in choice_labels}

        # majority ensemble
        majority_choices = {choice: 0 for choice in choice_labels}
        for i, model_name in enumerate(selected_models):
            predictions = models_subject_test_predictions[model_name]
            if idx >= len(predictions):
                continue  # Skip if prediction index is out of range
            prediction = predictions[idx]
            predicted_answer = prediction.get('predicted_answer', None)
            if predicted_answer in choice_probs:
                choice_probs[predicted_answer] += 1
        final_majority_answer = max(majority_choices, key=majority_choices.get)
        if final_majority_answer == correct_answer:
            mvoting_correct += 1

        # DFPE ensemble
        for i, model_name in enumerate(selected_models):
            weight = weights[i]
            if weight == 0:
                continue  # Skip models with zero weight
            predictions = models_subject_test_predictions[model_name]
            if idx >= len(predictions):
                continue  # Skip if prediction index is out of range
            prediction = predictions[idx]
            predicted_answer = prediction.get('predicted_answer', None)
            if predicted_answer in choice_probs:
                choice_probs[predicted_answer] += weight

        # Normalize probabilities
        total_prob = sum(choice_probs.values())
        if total_prob > 0:
            for choice in choice_probs:
                choice_probs[choice] /= total_prob
        else:
            # Assign uniform probability if no weights were added
            for choice in choice_probs:
                choice_probs[choice] = 1.0 / len(choice_probs)

        # Determine final answer
        final_answer = max(choice_probs, key=choice_probs.get)

        # Update metrics
        if final_answer == correct_answer:
            dfpe_correct += 1
        total += 1

    return dfpe_correct, mvoting_correct, total


def perform_ensemble(models_test_predictions,
                     model_accuracies,
                     models_fingerprint,
                     args
                     ):
    # Get list of subjects
    subjects = set()
    for model_name in models_fingerprint:
        subjects.update(models_fingerprint[model_name].keys())

    cluster_labels = cluster_models_per_subject(models_fingerprint, subjects, esp=args.get_param("dbscan_epsilon"))
    def print_highest_cluster_number(cluster_labels):
        for subject, models in cluster_labels.items():
            highest_cluster_number = max(models.values())
            print(f"{subject}: {highest_cluster_number}")

    # Call the function with the cluster_labels
    print_highest_cluster_number(cluster_labels)

    mean_accuracies = calculate_mean_accuracy_per_subject(model_accuracies, subjects)
    print("Mean accuracy per subject:")
    for subject in subjects:
        print(f"Subject: {subject}, Mean Accuracy: {mean_accuracies[subject]:.4f}")

    total_dfpe_correct = 0
    total_mvoting_correct = 0
    total_questions = 0

    # Initialize a dictionary to store per-subject results
    per_subject_results = {}

    for subject in subjects:
        print(f"\nEvaluating subject: {subject}")
        # Select representative models
        selected_models = select_representative_models(subject, cluster_labels, model_accuracies)
        print(f"Selected models for subject '{subject}': {selected_models}")


        models_subject_test_predictions = {model: value[subject] for model, value in models_test_predictions.items() if model in selected_models}
        models_subject_accuracies = {model: value[subject] for model, value in model_accuracies.items() if model in selected_models}
        # Perform ensemble predictions on test set with scaling_factor and quantile_param
        dfpe_correct, mvoting_correct, total = ensemble_predictions_on_test_set(
            selected_models,
            models_subject_test_predictions,
            models_subject_accuracies,
            quantile_param=args.get_param("quantile_threshold"),
            scaling_factor=args.get_param("scaling_factor")
        )

        # Calculate accuracy
        dfpe_subject_accuracy = dfpe_correct / total if total > 0 else 0
        mvoting_subject_accuracy = mvoting_correct / total if total > 0 else 0

        total_dfpe_correct += dfpe_correct
        total_mvoting_correct += mvoting_correct
        total_questions += total

        # Store per-subject results
        per_subject_results[subject] = {
            "selected_models": selected_models,
            "dfpe_correct": dfpe_correct,
            "mvoting_correct": mvoting_correct,
            "total": total,
            "dfpe_accuracy": dfpe_subject_accuracy,
            "mvoting_accuracy": mvoting_subject_accuracy
        }

    # Print Subject Accuracy
    print("MVoting Subject-Accuracy: ")
    print({subject:vals['mvoting_accuracy'] for subject,vals in per_subject_results.items()})

    print("DFPE Subject-Accuracy: ")
    print({subject: vals['dfpe_accuracy'] for subject, vals in per_subject_results.items()})


    # Calculate overall accuracy
    dfpe_overall_accuracy = total_dfpe_correct / total_questions if total_questions > 0 else 0
    mvoting_overall_accuracy = total_mvoting_correct / total_questions if total_questions > 0 else 0

    # Calculate subject accuracy average
    dfpe_subject_accuracy_avg = np.mean([i['dfpe_accuracy'] for i in per_subject_results.values()])
    mvoting_subject_accuracy_avg = np.mean([i['mvoting_accuracy'] for i in per_subject_results.values()])

    print(f"MVoting Overall-Accuracy: {mvoting_overall_accuracy:.3}")
    print(f"DFPE Overall-Accuracy: {dfpe_overall_accuracy:.3}")

    print(f"MVoting Subject-Accuracy Average: {mvoting_subject_accuracy_avg:.3}")
    print(f"DFPE Subject-Accuracy Average : {dfpe_subject_accuracy_avg:.3}")