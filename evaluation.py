import re
import string
from collections import Counter


#def remove_articles(text):
#        return re.sub(r"\b(a|an|the|An|A|The|THE|AN|is|in|and|s)\b", " ", text)
#
#def normalize_answer_cased(s):
#    def remove_articles(text):
#        return re.sub(r"\b(a|an|the|An|A|The|THE|AN|is|in|and|s)\b", " ", text)
#
#    def white_space_fix(text):
#        return " ".join(text.split())
#
#    def remove_punc(text):
#        exclude = set(string.punctuation+"‘’´`_")
#        return "".join(ch for ch in text if ch not in exclude)
#
#    return white_space_fix(remove_articles(remove_punc(s)))

#def normalize_answer(s):
#    def remove_articles(text):
#        return re.sub(r"\b(a|an|the)\b", " ", text)
#
#    def white_space_fix(text):
#        return " ".join(text.split())
#
#    def remove_punc(text):
#        exclude = set(string.punctuation+"‘’´`_")
#        return "".join(ch for ch in text if ch not in exclude)
#
#    def lower(text):
#        return text.lower()
#
#    def replace_underscore(text):
#        return text.replace('_', ' ')
#
#    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

#def normalize_answer(s):
#    """Lower text and remove punctuation, articles and extra whitespace."""
#
#    def remove_articles(text):
#        return re.sub(r'\b(a|an|the)\b', ' ', text)
#
#    def white_space_fix(text):
#        return ' '.join(text.split())
#
#    def handle_punc(text):
#        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
#        return ''.join(ch if ch not in exclude else ' ' for ch in text)
#
#    def lower(text):
#        return text.lower()
#
#    def replace_underscore(text):
#        return text.replace('_', ' ')
#
#    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

#def normalize_answer_w_articles(s):
#    def remove_articles(text):
#        return re.sub(r"\b(a|an|the)\b", " ", text)
#
#    def white_space_fix(text):
#        return " ".join(text.split())
#
#    def remove_punc(text):
#        exclude = set(string.punctuation+"‘’´`_")
#        return "".join(ch for ch in text if ch not in exclude)
#
#    def lower(text):
#        return text.lower()
#
#    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def nq_ems(prediction, answers):
    prediction = list(filter(None, prediction.split('answer: ')))
    prediction_set = set([normalize_answer(p) for p in prediction])
    gold_set = set([normalize_answer(g) for g in answers])
    return int(prediction_set == gold_set)