from arguments_manager import ArgumentsManager
from dataset import Dataset
import inference
from sentence_transformers import SentenceTransformer
from model import Model
import ensemble


def main():
    # parse arguments
    args = ArgumentsManager()
    args.parse_args()

    # prepare dataset
    dataset = Dataset(
        dataset_path=args.get_param("hf-dataset-name"),
        question_column_name=args.get_param("question-column"),
        choices_column_name=args.get_param("choices-column"),
        subject_column_name=args.get_param("subject-column"),
        label_column_name=args.get_param("label-column")
    )

    # prepare embedding model
    embedding_model = SentenceTransformer(args.get_param("embedding-model"))

    # TODO: parallel
    # models process
    models_val_predictions = {}
    models_test_predictions = {}
    models_accuracies = {}
    models_fingerprint = {}
    for model_name in args.get_param("models"):
        model = Model(model_name)
        val_predictions, test_predictions, accuracy, fingerprint = inference.process_model(dataset, model, embedding_model, batch_size=args.get_param("batch-size"))

        models_val_predictions[model_name] = val_predictions
        models_test_predictions[model_name] = test_predictions
        models_accuracies[model_name] = accuracy
        models_fingerprint[model_name] = fingerprint


    # ensemble results
    ensemble.perform_ensemble(models_test_predictions, models_accuracies, models_fingerprint, args)

if __name__ == "__main__":
    main()