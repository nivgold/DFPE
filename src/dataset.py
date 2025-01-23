from collections import defaultdict
from datasets import load_dataset
import utils


class Dataset:
    def __init__(self,
                 dataset_path: str,
                 question_column_name: str,
                 choices_column_name: str,
                 subject_column_name: str,
                 label_column_name: str
                 ):

        dataset = load_dataset(dataset_path, 'all')
        self.question_column_name = question_column_name
        self.choices_column_name = choices_column_name
        self.subject_column_name = subject_column_name
        self.label_column_name = label_column_name

        self.validation_data = self.preprocess(dataset['validation'])
        self.test_data = self.preprocess(dataset['test'])
        self.subjects = set([item["category"] for item in self.validation_data])

        # Organize validation and test questions by subject
        self.validation_subject_questions = defaultdict(list)
        self.test_subject_questions = defaultdict(list)
        for example in self.validation_data:
            subject = example[self.subject_column_name]
            self.validation_subject_questions[subject].append(example)
        for example in self.test_data:
            subject = example[self.subject_column_name]
            self.test_subject_questions[subject].append(example)

    def preprocess(self, dataset):
        res = []
        for each in dataset:
            question = each[self.question_column_name]
            options = each[self.choices_column_name]
            correct_answer_index = int(each[self.label_column_name])
            category = each[self.subject_column_name]
            res.append({
                "question": question,
                "prompt": utils.format_prompt(question, options),
                "options": options,
                "answer_index": correct_answer_index,
                "correct_answer": ['A', 'B', 'C', 'D'][correct_answer_index],
                "category": category
            })
        return res