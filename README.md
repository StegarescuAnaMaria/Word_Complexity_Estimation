# Word Complexity Estimation

1. Feature Extraction
For feature extraction, I processed the sentences by transforming them to lowercase and getting 
rid of punctuation, then separating the sentences to lists of words. I chose 2 methods of feature 
extraction – the classic bag of words and tf-idf (the comparison will be discussed further).
I tried to experiment with both word stemming and tokenization for the bag of words features, 
which brought negative results on the test data compared to the usual bag of words. 
I initially tested the different methods of feature extraction with MLPRegressor from the sklearn 
library, with SGD optimizer, 200 epochs and hidden layer size of (100,), and ReLU 
In this documentation, I will refer to the private Kaggle scores, estimated on 25% of the data, as 
it was the only score I could take into consideration while working on the project. 
The MLPRegressor, with the usual bag of words features, gave me a score of 0.05839 MSE. The 
same model, with stemming, gave me a score of 0.089 MSE, and with word tokenization –
0.06088 MSE. I have theorized upon the bad results of such word extraction practices for the 
word complexity estimation: stemming removes important word parts that are necessary for the 
experiment, like the ‘-ing’ from a verb or the ‘-s’ from the plural of a noun, and word 
tokenization doesn’t separate compound words united through a dash, like ‘long-term’; if ‘long-term’ is recognized as a single word, there is a high chance that its appearance in the data will be 
sparse, and the model will not be able to apply its learned complexity from the train data to the 
test data, or will not be able to predict its complexity in case it appeared only in the test data and 
not in the train data.
TF-IDF vs Bag-of-Words:
TF-IDF proved to give better results (0.05102 MSE), with the same MLPRegressor model with 
same parameters as above. I theorize that it is mostly because of TF, as it is equal to 1 divided to 
the number of words in the phrase. In most cases, the complexity grows parallel to the number of 
words in the phrase. Let’s take 2 examples from the train data: ‘coral’, with 0.0 complexity, and 
‘coral outcrops’, with 0.1 complexity. Of course, it is not entirely accurate for all examples. I 
think IDF didn’t make a big change in the results. It focuses more on the relevance of a word to 
the rest of the text, by counting the number of appearances of it in the corpus of documents.
While I believe it is a very useful practice in most NLP experiments, I think that the word 
complexity estimation is an exception. Whether it appears frequently or not in other sentences, 
most likely, doesn’t play a role to the complexity perceived by the annotators.

