__all__ = ["LimeExplainer"]

from explanation.base import BaseExplainer

from lime.lime_text import LimeTextExplainer
import re

class LimeExplainer(BaseExplainer):
    def __init__(self, classifier):
        super().__init__(classifier)
        self.explainer = LimeTextExplainer(split_expression=self.split_rule, class_names=list(self.ind2label.values()))

    def split_rule(self, text):
        sentences = re.split('(。|！|\!|\.|？|\?)',text)
        new_sents = []
        for i in range(int(len(sentences)/2)):
            sent = sentences[2*i] + sentences[2*i+1]
            new_sents.append(sent)
        return new_sents

    def explain(self, text, top3_author_ids):
        if(len(self.split_rule(text)) <= 1):
            return text.replace(' ', ''), None
        
        exp = self.explainer.explain_instance(text, self.predict_proba, num_features=1000, top_labels=10)
        exp_map = exp.as_map()
        
        text = text.replace(' ', '')
        sentences = self.split_rule(text)
        
        newd = {}
        for sent_index, value in exp_map.items():
            newd[sent_index] = [t for t in exp_map[sent_index] if t[1] > 0.01]
        exp_map = newd

        result = {}
        for author_id, tuples in newd.items():
            for sent_index, score in tuples:
                sent_index = int(sent_index)
                sentence = sentences[sent_index]
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                score = round(score * 100, 2)

                if sent_index not in result:
                    result[sent_index] = [(start_pos, end_pos)]
                result[sent_index].append([author_id, score])

        for key, value in result.items():
            positions, *authors_scores = value
            authors_scores = sorted(authors_scores, key=lambda x: x[1], reverse=True)
            try:
                rank = top3_author_ids.index(int(authors_scores[0][0]))
                color = self.first_three_colors[rank]
            except ValueError:
                color = '#212F3D'
            result[key] = [positions, color, authors_scores] # sort by score

        for i in range(len(sentences)): # Sentences not similar to any author
            if i not in result:
                sentence = sentences[i]
                start_pos = text.find(sentence)
                end_pos = start_pos + len(sentence)
                result[i] = [(start_pos, end_pos), '#212F3D', [[-1, 0]]]
        
        result = dict(sorted(result.items(), key=lambda item: item[1][0][0])) # sort by sentence position
        return text, result
    
