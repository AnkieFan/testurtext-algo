__all__ = ["ShapExplainer"]

from explanation.base import BaseExplainer

import shap
import re
from typing import List, Dict

class CustomTokenizer:
    def __init__(self):
        self.vocab = {}
        self.ids_to_tokens = {}
        self.tokenizer_id = 0
    
    def split_rule(self, text: str) -> List[str]:
        sentences = re.split('(。|！|\!|\.|？|\?)', text)
        new_sents = []
        for i in range(int(len(sentences)/2)):
            sent = sentences[2*i] + sentences[2*i+1]
            new_sents.append(sent)
        return new_sents
    
    def tokenize(self, text: str) -> List[str]:
        tokens = self.split_rule(text)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.tokenizer_id
                self.ids_to_tokens[self.tokenizer_id] = token
                self.tokenizer_id += 1
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[token] for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens[i] for i in ids]
    
    def encode(self, text: str) -> Dict[str, List[int]]:
        tokens = self.tokenize(text)
        input_ids = self.convert_tokens_to_ids(tokens)
        offset_mapping = [(m.start(), m.end()) for m in re.finditer('|'.join(map(re.escape, tokens)), text)]
        return {
            'input_ids': input_ids,
            'offset_mapping': offset_mapping
        }
    
    def decode(self, input_ids: List[int]) -> str:
        tokens = self.convert_ids_to_tokens(input_ids)
        return ''.join(tokens)
    
    def __call__(self, text: str) -> Dict[str, List[int]]:
        return self.encode(text)

class ShapExplainer(BaseExplainer):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.tokenizer = CustomTokenizer()
        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(self.predict_proba, masker, output_names = list(self.ind2label.values()))
        
    def explain(self, text, top3_author_ids):
        if(len(self.tokenizer.split_rule(text)) <= 1):
            return text.replace(' ', ''), None
        
        exp = self.explainer([text]).values[0]
        
        text = text.replace(' ', '')
        sentences = self.tokenizer.split_rule(text)

        result = {}
        for i in range(len(exp)):
            sentence = sentences[i]
            start_pos = text.find(sentence)
            end_pos = start_pos + len(sentence)

            indexed_list = list(enumerate(exp[i]))
            sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
            top_3_indexes = [(author_id, round(score * 100, 2)) for author_id, score in sorted_indexed_list[:3] if score > 0.001]

            try:
                rank = top3_author_ids.index(int(top_3_indexes[0][0]))
                color = self.first_three_colors[rank]
            except ValueError:
                color = '#212F3D'

            result[i] = [(start_pos, end_pos), color, top_3_indexes]
            
        return text, result
