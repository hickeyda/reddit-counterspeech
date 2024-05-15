Repository containing data for the paper "Hostile Counterspeech Drives Users from Hate Subreddits"

### Description of files

data/subreddits_used_for_analysis.csv: A description of the hate subreddits used in the study.

data/counterspeech_dataset.csv: Labeled examples of counterspeech used to train the counterspeech detection model. Further details can be found in datasheet_for_counterspeech_dataset.md

data/all_subreddits_hate_lexicon_with_replacement.csv: Hate words obtained via SAGE from the 25 subreddits, used for the word replacement analysis.

code/embed_counterspeech.py: Code to embed the text from the counterspeech dataset and compile the features into a Pytorch dataset.

code/process_yu_et_al.py: Code to embed the text from the counterspeech dataset provided by Yu et al. Their dataset can be found at: https://github.com/xinchenyu/counter_context

code/train_counterspeech_model.py: Code to train a neural network to detect counterspeech from text embeddings and post metadata. Set ADDITIONAL_FEATURES = False to exclude the features of the subreddits and karma and set YU_ET_AL = True to train on the dataset provided by Yu et al.

code/evaluate_performance.py: Code that takes files produced by train_counterspeech_model.py and reports the average ROC-AUC score and average F1, recall, and precision scores for the threshold that maximizes the F1 score.
