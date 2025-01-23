import argparse


class DefaultArguments:
    DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    DEFAULT_DBSCAN_EPSILON = 0.5
    DEFAULT_QUANTILE_FILTER_THRESHOLD = 0.95
    DEFAULT_SCALING_FACTOR = 1.0
    DEFAULT_BATCH_SIZE = 1



class ArgumentsManager:
    def __init__(self):
        self.parsed_args = None
        parser = argparse.ArgumentParser(description="DFPE options")

        # models to use
        parser.add_argument("--models", "-m", type=lambda x: [i.strip() for i in x.split(",")], nargs='+', required=True,
                            help="LLM Models list separated by comma")

        # HF dataset name
        parser.add_argument("--hf-dataset-name", "-hd", type=str,
                            help="HF dataset name")

        # question column name
        parser.add_argument("--question-column", "-q", type=str, required=True,
                            help="Name of the question column in the dataset")

        # choices column name
        parser.add_argument("--choices-column", "-c", type=str, required=True,
                            help="Name of the choices column in the dataset")

        # subject column name
        parser.add_argument("--subject-column", "-cat", type=str, required=True,
                            help="Name of the subject column in the dataset")

        # label column name
        parser.add_argument("--label-column", "-l", type=str, required=True,
                            help="Name of the label column in the dataset")

        # embedding model to use
        parser.add_argument("--embedding-model", "-e", type=str, required=True,
                            default=DefaultArguments.DEFAULT_EMBEDDING_MODEL_NAME,
                            help="Embedding model to use for encoding the dataset")

        # DBSCAN epsilon
        parser.add_argument("--dbscan-epsilon", "-eps", type=float,
                            default=DefaultArguments.DEFAULT_DBSCAN_EPSILON,
                            help="DBSCAN epsilon value for clustering")

        # quantile filter threshold
        parser.add_argument("--quantile-threshold", "-qt", type=float,
                            default=DefaultArguments.DEFAULT_QUANTILE_FILTER_THRESHOLD,
                            help="Quantile threshold for filtering data")

        # scaling factor
        parser.add_argument("--scaling-factor", "-s", type=float,
                            default=DefaultArguments.DEFAULT_SCALING_FACTOR,
                            help="Scaling factor for normalization or transformation")

        # scaling factor
        parser.add_argument("--batch-size", "-b", type=float,
                            default=DefaultArguments.DEFAULT_BATCH_SIZE,
                            help="batch size of the models inference")

        self.parser = parser

    def parse_args(self):
        parsed_args = self.parser.parse_args()
        self.parsed_args = parsed_args.__dict__

    def get_param(self, param_name):
        return self.parsed_args.get(param_name, None)