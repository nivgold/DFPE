from arguments_manager import ArgumentsManager
from dataset import Dataset
import inference
from sentence_transformers import SentenceTransformer
from model import Model
import ensemble
import utils


def main():
    # parse arguments
    args = ArgumentsManager()
    args.parse_args()

    # prepare dataset
    dataset = Dataset(
        dataset_path=args.get_param("hf_dataset_name"),
        question_column_name=args.get_param("question_column"),
        choices_column_name=args.get_param("choices_column"),
        subject_column_name=args.get_param("subject_column"),
        label_column_name=args.get_param("label_column")
    )

    # prepare embedding model
    embedding_model = SentenceTransformer(args.get_param("embedding_model"))

    # TODO: parallel
    # models process
    models_val_predictions = {}
    models_test_predictions = {}
    models_accuracies = {}
    models_fingerprint = {}
    for model_name in args.get_param("models"):
        model = Model(model_name)
        val_predictions, test_predictions, accuracy, fingerprint = inference.process_model(dataset, model, embedding_model, batch_size=args.get_param("batch_size"))

        models_val_predictions[model_name] = val_predictions
        models_test_predictions[model_name] = test_predictions
        models_accuracies[model_name] = accuracy
        models_fingerprint[model_name] = fingerprint

    bsmov_overall_acc, bsmov_subject_acc = utils.calculate_best_single_metrics(models_val_predictions)
    bsm_overall_acc, bsm_subject_acc = utils.calculate_best_single_metrics(models_test_predictions)
    print("BSM Overall-Accuracy: ")
    print(bsm_overall_acc)
    print("BSM Subject-Accuracy: ")
    print(bsm_subject_acc)

    print("BSMoV Overall-Accuracy")
    print(bsmov_overall_acc)
    print("BSMoV Subject-Accuracy")
    print(bsmov_subject_acc)


    # ensemble results
    ensemble.perform_ensemble(models_test_predictions, models_accuracies, models_fingerprint, args)

if __name__ == "__main__":
    main()