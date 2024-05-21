import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification
from dotenv import load_dotenv

from CountryGenderNamePerturbation import PerturbedExamples
from utils import create_input_array, symetric_kl

load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR", None)
PATH_DATA = os.getenv("PATH_DATA", None)

class BiasesCalculator:
    def __init__(self, model_name, path_corpus,
                 data_tsv, text_col, label_col,
                 list_countries, n_duplicates, max_seq_len_ini=256):
        self.model_name = model_name
        self.path_corpus = path_corpus
        self.data_tsv = data_tsv
        self.text_col = text_col
        self.label_col = label_col
        self.list_countries = list_countries
        self.n_duplicates = n_duplicates
        self.max_seq_len_ini = max_seq_len_ini
        self.tokenizer = None
        self.model = None
        self.config = None
        self.dict_lab = None
        self.tsv_path = os.path.join(self.path_corpus, self.data_tsv)

        self.load_model_tokenizer_and_config()
    
    def load_model_tokenizer_and_config(
            self,
            model_gen=TFAutoModelForSequenceClassification,
            tok_gen=AutoTokenizer):
        tokenizer = tok_gen.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        )
        model = model_gen.from_pretrained(
            self.model_name, cache_dir=CACHE_DIR
        )

        config = AutoConfig.from_pretrained(self.model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dict_lab = {k.lower(): v for k, v in config.label2id.items()} # Sometimes this is wrong

        # for sentiment en model there was an issue
        # if 'label_0' in self.dict_lab.keys(): 
            # self.dict_lab = {'negative': 0, 'neutral': 1, 'positive': 2}
    

    def read_tsv_to_inputs_data(self, encoding="utf-8"):
        tsv_df = pd.read_csv(
        self.tsv_path, sep="\t", quotechar='"', encoding=encoding, header=0
        )
        text_data = tsv_df[self.text_col].values
        labels_data = tsv_df[self.label_col].values
        df = pd.DataFrame({"tweet": text_data, "label": labels_data})

        train_tweets = df["tweet"].fillna("CVxTz").values # random values, never happens

        if df["label"].str.lower().map(dict_lab).isnull().sum() != 0:
            print(dict_lab)
            print(df["label"].str.lower().map(dict_lab).isnull().sum())
            list_unique = df["label"].unique()
            print(list_unique)
            if (len(list_unique) != 1) or (list_unique[0] != "-1"):
                raise Exception(
                    "Error: dict_lab not working well with the dataset"
                )
            else:
                print("Dataset of unlabeled data...ok!")

        train_y = (
            df["label"].str.lower().map(dict_lab).astype(np.int32).values
        )

        if len(dict_lab) != len(df["label"].unique()):
            print("There might be a problem with the labels...")
            print(
                f'dict_lab : {dict_lab},\ndf["label"].unique() : {df["label"].unique()}\n\n'
            )

        return (train_tweets, train_y)
    
    def _calculate_sentiment_bias(
        self,
        X_text,
        y,
        dict_pos_neg={"positive": "yes", "negative": "no"},
    ):
        """
        The function itself, taking model, X_text, y as inputs
        """

        str_pos = "positive"
        str_neg = "negative"
        # if not positive negative, then you need a mapping between the variables and pos/neg (like for stance rec)
        if not (("positive" in self.dict_lab.keys()) and ("negative" in self.dict_lab.keys())):
            # check that all the labels are in the dict_pos_neg so that can be a direct mapping
            for k in dict_pos_neg.values():
                assert k in list(
                    self.dict_lab.keys()
                ), "dict_pos_neg does not contain the string names of positive/negative classes"

            print("new mapping:", dict_pos_neg)
            str_pos = dict_pos_neg[str_pos]
            str_neg = dict_pos_neg[str_neg]


        perturber = (
            PerturbedExamples(self.list_countries)
            if len(self.list_countries)
            else PerturbedExamples()
        )
        print("Perturbing examples...")
        perturbed_X_text = perturber.all_countries(X_text, y, self.n_duplicates)
        nb_sent_modified = len(perturbed_X_text["Original"][0])
        print("...Obtaining %d perturbated sentences" % nb_sent_modified)
        dict_results = {country: {} for country in perturbed_X_text.keys()}

        path_dump_perturbed = os.path.join(self.path_corpus, "perturbed_text_stances.pkl")
        with open(path_dump_perturbed, "wb") as fp:
            pkl.dump(perturbed_X_text, fp)

        if nb_sent_modified > 1:
            print("Calculating probas...")
            # print(perturbed_X_text['Original'][0])

            input_arrays = create_input_array(perturbed_X_text["Original"][0], self.tokenizer)
            try:  # for TFAutoModelForSequenceClassification
                proba_ini = softmax(self.model.predict(input_arrays), axis=1)
            except:
                proba_ini = softmax(self.model.predict(input_arrays)[0], axis=1)
            y_hat_ini = [
                kkk
                for kk in [self.n_duplicates * [k] for k in np.argmax(proba_ini, axis=1)]
                for kkk in kk
            ]

            # mean of the difference between pos and neg
            dict_results["Original"] = 100 * np.mean(
                proba_ini[:, self.dict_lab[str_pos]] - proba_ini[:, self.dict_lab[str_neg]]
            )

            for country, value_gender in tqdm(perturbed_X_text.items()):
                if country != "Original":
                    for gender, (examples, labels) in value_gender.items():
                        input_arrays = create_input_array(examples, self.tokenizer)
                        # print(model.predict(input_arrays))
                        try:
                            proba = softmax(self.model.predict(input_arrays), axis=1)
                        except:  # for TFAutoModelForSequenceClassification
                            proba = softmax(self.model.predict(input_arrays)[0], axis=1)
                        dict_results[country][gender] = {
                            "proba": np.round(
                                100
                                * np.mean(
                                    proba[:, self.dict_lab[str_pos]]
                                    - proba[:, self.dict_lab[str_neg]]
                                )
                                - dict_results["Original"],
                                2,
                            )
                        }
                        y_hat_pred = np.argmax(proba, axis=1)
                        CM = confusion_matrix(y_hat_ini, y_hat_pred)

                        dump_CM_file = f"CM_stances_en_{country}_{gender}.pkl"
                        path_dump_CM = os.path.join(self.path_corpus, dump_CM_file)
                        with open(path_dump_CM, "wb") as fp:
                            pkl.dump(CM, fp)
                        # gives the percentage of increasing/decreasing
                        # the number of new prediction - the number of past prediction / the number of past prediction
                        percentage_changing = (
                            np.sum(CM, axis=0) - np.sum(CM, axis=1)
                        ) / np.sum(CM, axis=1)

                        # case where there was only one class outputed by the system, i.e. if very few entities were detected
                        if len(CM) == 1:
                            print(percentage_changing)
                            percentage_changing = [1 for _ in self.dict_lab.values()]

                        # print(y_hat_ini, y_hat_pred, CM)
                        for lab, idx_lab in self.dict_lab.items():
                            dict_results[country][gender][lab] = 100 * np.round(
                                percentage_changing[idx_lab], 2
                            )

                        dict_results[country][gender]["KL_sym"] = symetric_kl(
                            np.repeat(proba_ini, self.n_duplicates, axis=0), 
                            proba,
                            mean_of_divs=True,
                        )
                        dict_results[country][gender]["KL_sym_mean"] = symetric_kl(
                            np.repeat(proba_ini, self.n_duplicates, axis=0),
                            proba,
                            mean_of_divs=False,
                        )

            # creating the df
            del dict_results["Original"]
            midx = pd.MultiIndex.from_product(
                [
                    list(dict_results.values())[0].keys(),
                    list(list(dict_results.values())[0].values())[0].keys(),
                ]
            )
            df_bias = pd.DataFrame(
                [
                    [key3 for key2 in key1.values() for key3 in key2.values()]
                    for key1 in dict_results.values()
                ],
                index=dict_results.keys(),
                columns=midx,
            )

        else:
            print("Not even one sentence detected with an entity... Return empty dataframe")
            df_bias = pd.DataFrame([[]])

        df_bias["n_sent_modified"] = nb_sent_modified
        return df_bias
