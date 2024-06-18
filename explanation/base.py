# -*- coding: utf-8 -*-

import numpy as np
import re
import json
import logging
#logging.basicConfig(level=logging.DEBUG)
from abc import ABC

class BaseExplainer(ABC):
    def __init__(self, classifier):
        self.classifier = classifier
        label2ind = json.load(open('data/files/label2ind.json', 'r', encoding='utf-8'))
        self.ind2label = {value: key for key, value in label2ind.items()}
        self.first_three_colors = ['#fead57', '#82aab8', '#92363b']

    def predict_proba(self, texts):
        texts = list(texts)
        labels, probabilities = self.classifier.predict(texts, k=len(self.classifier.get_labels()))
        mapped_labels = [[self.ind2label[label] for label in label_list] for label_list in labels]
        mapped_probabilities = [dict(zip(mapped_label_list, prob_list)) for mapped_label_list, prob_list in zip(mapped_labels, probabilities)]
        return np.array([[prob_dict.get(class_name, 0) for class_name in self.ind2label.values()] for prob_dict in mapped_probabilities])

    @classmethod
    def explain(self, text, top3_author_ids):
        """
        Args:
            text: str

        Returns:
            text: str 
                This text is preprocessed, might be different from the user's original input. Please use this text to display the result.
            result: { sentence_index: [(start, end), color(dark grey as default), *(authorID, percentage)] }
        """
