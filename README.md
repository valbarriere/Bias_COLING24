# AI Xenophobia 

Code of the paper [Are Text Classifiers Xenophobic? A Country-Oriented Bias Detection Method with Least Confounding Variables](https://aclanthology.org/2024.lrec-main.134/), Barriere, V., & Cifuentes, S., In *Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)* (pp. 1511-1518). 

The goal of this repo is to give a score that represent the bias of a production model, using unlabeled production data, toward different countries, using the names as proxy. We found out multilingual models based on XLM-R tend to favour names coming from a country speaking the language of the sentence. Please cite our paper if you use the toolbox. 

## Requirements

- Python version == 3.10

### CheckList library

This repository uses a custom version of the [CheckList library](https://github.com/marcotcr/checklist)

### Required packages

For managing Python packages, it is recommended to use either `virtualenv` or `conda`.

- For `virtualenv`, you can create a virtual environment using the following commands:

```bash
pip install virtualenv
virtualenv biases-ppl
source biases-ppl/bin/activate
```

- For `conda`, you can create a conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate biases-ppl
```

The required packages are listed in the `requirements.txt` file. You can install them using `pip`.

```bash
pip install -r requirements.txt
```

### Environment file

We use a `.env` file to store environment variables. Please copy the template from `public.env` to `.env` and fill in the required values.

```bash
cp public.env .env
```

### Entity Recognition Model

To run the experiments we use an entity recognition model. Download it using the Spacy library:

```bash
python -m spacy download xx_ent_wiki_sm
```

### XLM-T data

We use the tweets available in the repository of XLM-T. Clone the repo and generate the data in tsv format:

```bash
git clone git@github.com:cardiffnlp/xlm-t.git
python build_xlmt_tsv_data.py
```

## Run the experiments


### One-liners

To perturb the data and calculate the bias without calculating perplexity, run:

```bash
python biases_calculation.py \
--name_corpora tsv_data_xlmt \
--data_tsv tweets_test_english.tsv \
--label_type int \
--list_countries France United_Kingdom Spain Germany Italy Morocco \
Portugal Hungary Poland Turkey \
--n_duplicates 50 \
--model_name cardiffnlp/twitter-xlm-roberta-base-sentiment
```

This command will create a new file inside the input folder with "biases" as prefix. In the example, the new file will be "biases_tweets_test_english.tsv".

After running the biases calculation over the tweets, you can build the confusion matrix for the data using the following command:

```bash
python build_confusion_matrix.py tsv_data_xlmt "biases_tweets_test_{}.tsv"
```

Where the first argument is the folder containing the biases calculation and the second argument is the pattern of the biases file names


### Class

You can use the class `BiasesCalculator` from `biases_calculator.py` in order to calculate the bias for a specific dataset and model, like in the `biases_calculation.py` script. 

```python
biases_calculator = BiasesCalculator(args.model_name,
                                     path_corpus,
                                     args.data_tsv,
                                     args.text_col,
                                     args.label_col,
                                     args.list_countries,
                                     args.n_duplicates)

X_text, y = biases_calculator.read_tsv_to_inputs_data() # Reads the data and loads it as tweet, label

df_bias = biases_calculator._calculate_sentiment_bias(
    X_text,
    y
)
print(df_bias)
```
