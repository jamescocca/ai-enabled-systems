# Caroline Cocca
# February 20, 2022
# Creating AI-Enabled Systems
# Assignment 3
import pandas as pd
import nltk


# read in the summary col, convert to string, and return
def get_summary(path):
    reviews_df = pd.read_csv(path)
    summary = reviews_df["summary"].tolist()
    summary = "".join(summary)
    return summary


# take in a string and perform word, casual, and sentence tokenization
# then print and return the 3 token lists
def do_tokens(start_string):
    wtokens = nltk.word_tokenize(start_string)
    print("First 10 summary word tokens:", wtokens[:10])
    ctokens = nltk.casual_tokenize(start_string)
    print("First 10 summary casual tokens:", ctokens[:10])
    stokens = nltk.sent_tokenize(start_string)
    print("First 5 summary sentence tokens:", stokens[:5])
    print()
    return wtokens, ctokens, stokens


# perform Porter, Lancaster, and Snowball stemming on the given tokens
# print the first 10 stems for each result
def do_stems(tokens):
    pstems = []
    lstems = []
    sstems = []
    for token in tokens:
        pstems.append(nltk.stem.porter.PorterStemmer().stem(token))
        lstems.append(nltk.stem.lancaster.LancasterStemmer().stem(token))
        sstems.append(nltk.stem.SnowballStemmer("english").stem(token))

    pstems_tokens = dict(zip(tokens[:10], pstems[:10]))
    print("First 10 Porter stems:", pstems_tokens)
    lstems_tokens = dict(zip(tokens[:10], lstems[:10]))
    print("First 10 Lancaster stems:", lstems_tokens)
    sstems_tokens = dict(zip(tokens[:10], sstems[:10]))
    print("First 10 Snowball stems:", sstems_tokens)
    print()


# perform WordNet lemmatization on the given tokens
# print the first 10 results
def do_lemmas(tokens):
    lemmas = []
    for token in tokens:
        lemmas.append(nltk.stem.wordnet.WordNetLemmatizer().lemmatize(token))

    lemmas_tokens = dict(zip(tokens[:10], lemmas[:10]))
    print("First 10 WordNet lemmas:", lemmas_tokens)
    print()


if __name__ == '__main__':
    nltk.download('omw-1.4')
    summary = get_summary("archive/Musical_instruments_reviews.csv")
    wtokens, ctokens, stokens = do_tokens(summary)
    do_stems(wtokens)
    do_lemmas(wtokens)
