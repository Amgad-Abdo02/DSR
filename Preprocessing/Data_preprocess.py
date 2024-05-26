import pandas as pd
import nltk
import string # Used for cleaning punctuation
from gensim.parsing.preprocessing import remove_stopwords #used to remove stop words (unlike nltk it considers however as a stopword)
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup # Used for Removing HTML tags

#Downloading Word datasets used in lemmatization
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def text_preprocessing(text:str) -> str:

        """Applies Lemmatization and lowercase and removes stop words, Punctiouation, Qoutes 
            and HTML tags 

        Args: data -> (str)

        Returns:
            Cleaned text -> (str) : Preprocessed text for model.

        """
        lemmatizer = WordNetLemmatizer()

        temptxt = BeautifulSoup(text, 'html.parser').get_text()

        temptxt= temptxt.translate(str.maketrans('', '', string.punctuation)) 

        temptxt= temptxt.lower() 

        temptxt= remove_stopwords(temptxt)

        temptxt= temptxt.replace('"', '')

        word_list = word_tokenize(temptxt) 
        clean_words= ' '.join([lemmatizer.lemmatize(w) for w in word_list])

        return clean_words

def Dataset_Labeling(df: pd.DataFrame, Condition_col_name:str) -> None:
        """ Adds label coloumn in Dataframe that contains the label of each condition 

        Args: 
            df -> (Dataframe) : dataframe that contains dataset

            Condition_col_name -> (str) : Name of the Conditions column

        Returns:
            None

        """         
        value_pair = {}

        def labeling(condition: str) -> int:
            label=value_pair[condition]
            return label
    
        unique_labels = df[Condition_col_name].unique()

        for i in range(0,len(unique_labels)):
            value_pair[unique_labels[i]]=i

        df["label"]=df[Condition_col_name].apply(labeling)

 
        
        

