import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nltk
import re
import io
import base64
from twython import Twython
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from sklearn.decomposition import LatentDirichletAllocatio


# Twitter API credentials
ACCESS_TOKEN = "761441357315440640-suCCQJo6kuufi3PmcYUl2y9kNyYb8C0"
ACCESS_TOKEN_SECRET = "nN4nX0LhlUZHN31LLYU1neOxg7elvb4LIo9KkX7gMDMaN"
API_KEY = "oMlZlYVi6MerYj7SZzcYWvgVr"
API_SECRET_KEY = "OW8cYRS69LUQ1gD5rKULGi4QtuBoj0OX5hRyJI5HVBbzTLZzam"

# extra stop words
STOP_WORDS = ["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must',
              "n't", 'need', 'sha', 'wo', 'would', "ca", "na", "rt", "like",
              'u', 'get', 'got']

#logging details
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


class TweetDetective():

    def __init__(self, search_query=''):
        '''This class gives insight about a given search query
        by analyzing tweets from Twitter.  
        '''

        logger.info('instantiating the TweetDetective object')
        self.search_query = search_query
        self.tweets_df = pd.DataFrame()        

    def collect_tweets(self, search_query='', geocode=None, result_type='recent', 
                       num_of_page = 20, count = 100, since = None, until = None):
        '''Collects a number of tweets using Twitter standard search API and
        returns a list of dictionaries each representing a tweet.

        query: search query
        geocode: Returns tweets by users located within a given radius
                of the given lat/long. The parameter value is specified
                by " latitude,longitude,radius "
        result_type: Specifies what type of search results you would prefer to receive.
                    mixed : Include both popular and real time results in the response.
                    recent : return only the most recent results in the response
                    popular : return only the most popular results in the response.
        num_of_page: number of pages to collect.
        count: The number of tweets to return per page, up to a maximum of 100.
            Defaults to 15.
        since: Returns tweets created after the given date.
            Date should be formatted as YYYY-MM-DD.
            The search index has a 7-day limit.
        until: Returns tweets created before the given date.
            Date should be formatted as YYYY-MM-DD.
            The search index has a 7-day limit.
        since_id: Returns results with an ID greater than
                (that is, more recent than) the specified ID.
                There are limits to the number of Tweets which
                can be accessed through the API. If the limit of
                Tweets has occured since the since_id, the since_id
                will be forced to the oldest ID available.
        max_id: Returns results with an ID less than
                (that is, older than) or equal to the specified ID.
        include_entities: The entities node will not be included when set to false.
        '''

        logger.info('Collecting tweets using twitter api...')
        # Authentication
        try:
            twitter_obj = Twython(API_KEY, API_SECRET_KEY,
                              ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

            # Use Twitter standard API search
            tweet_result = twitter_obj.search(q=search_query, geocode=geocode,
                                            result_type=result_type, count=count,
                                            since=since, until=until,
                                            include_entities='true',
                                            tweet_mode='extended', lang='en')
        except Exception as e:
            logger.exception(e)
            return -1


        # In order to prevent redundant tweets explained here
        # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
        # instead of reading a timeline relative to the top of the list
        # (which changes frequently), an application should read the timeline
        # relative to the IDs of tweets it has already processed.
        tweets_list = tweet_result['statuses']
        i = 0  # num of iteration through each page
        rate_limit = 1  # There is a limit of 100 API calls in the hour
        while tweet_result['statuses'] and i < num_of_page:
            if rate_limit < 1:
                # Rate limit time out needs to be added here in order to
                # collect data exceeding available rate-limit
                print(str(rate_limit)+' Rate limit!')
                break
            max_id = tweet_result['statuses'][len(
                tweet_result['statuses']) - 1]['id']-1

            try:
                tweet_result_per_page = twitter_obj.search(q=search_query, geocode=geocode,
                                                           result_type=result_type,
                                                           count=count, since=since,
                                                           until=until,
                                                           include_entities='true',
                                                           tweet_mode='extended',
                                                           lang='en',
                                                           max_id=str(max_id))
            except Exception as e:
                logger.exception(e)
                return -1

            tweets_list += tweet_result_per_page['statuses']
            i += 1
            rate_limit = int(twitter_obj.get_lastfunction_header(
                'x-rate-limit-remaining'))

        return tweets_list

    def find_hashtags(self, tweet):
        '''Find the hashtags in a tweet.
        '''

        hashtags = []
        for i, term in enumerate(tweet):
            hashtags += term['text']+','
        return hashtags

    def make_dataframe(self, tweets_list, search_term):
        '''Gets the list of tweets and return it as a pandas DataFrame.
        '''
        
        logger.info('Creating dataframe from tweets...')
        self.tweets_df = pd.DataFrame()
        self.tweets_df['tweet_id'] = list(map(lambda tweet: tweet['id'],
                                tweets_list))
        self.tweets_df['user'] = list(map(lambda tweet: tweet['user']
                            ['screen_name'], tweets_list))
        self.tweets_df['time'] = list(
            map(lambda tweet: tweet['created_at'], tweets_list))
        self.tweets_df['tweet_text'] = list(
            map(lambda tweet: tweet['full_text'], tweets_list))
        self.tweets_df['location'] = list(
            map(lambda tweet: tweet['user']['location'], tweets_list))
        self.tweets_df['hashtags'] = list(
            map(lambda tweet: self.find_hashtags(tweet['entities']['hashtags']), tweets_list))
        self.tweets_df['search_term'] = list(map(lambda tweet: search_term if search_term.lower(
        ) in tweet['full_text'].lower() else None, tweets_list))

        return self.tweets_df

    def clean_tweet_text(self, tweet, user_flag=True, urls_flag=True,
                         punc_flag=True, number_flag=True,
                         special_char_flag=True,
                         stop_word_flag=False):
        '''Clean a tweet by performing the following.

        - Remove username
        - Remove urls
        - Remove all punctuation and special character
        - Remove all stopwords if flag is True
        - Returns a cleaned text
        '''

        # remove the user
        if user_flag:
            tweet=re.sub(r'@[w\w]+', ' ', tweet)

        # remove the urls
        if urls_flag:
            tweet=re.sub(
                r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', tweet)

        # replace the negations
        tweet=re.sub(r"n't", ' not', tweet)
        tweet=re.sub(r"N'T", ' NOT', tweet)

        # remove punctuations
        if punc_flag:
            tweet=re.sub(
                '[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’…”'']+', ' ', tweet)

        # remove numbers
        if number_flag:
            tweet=re.sub('([0-9]+)', '', tweet)

        # remove special characters
        if special_char_flag:
            tweet=re.sub(r'[^\w]', ' ', tweet)

        # remove double space
        tweet=re.sub('\s+', ' ', tweet)

        if stop_word_flag:
            tweet=' '.join([word for word in tweet.split() if word.lower(
            ) not in stopwords.words('english')+STOP_WORDS])

        return tweet

    def tokenize_tweet(self, tweet):
        '''Convert the normal text strings in to a list of tokens(words)
        '''
        # Tokenization
        return [word for word in tweet.split()]

    def create_bag_of_words(self, tweets, max_df=0.95, min_df=2,
                                max_features=None):
        '''Vectorize the tweets using bag of words.

        Return the vectorized/bag of words of tweets
        as well as the features' name.

        max_df: float in range [0.0, 1.0] or int, default=1.0
                When building the vocabulary ignore terms that
                have a document frequency strictly higher than
                the given threshold (corpus-specific stop words).
                If float, the parameter represents a proportion
                of documents, integer absolute counts.
                This parameter is ignored if vocabulary is not None.

        min_df: float in range [0.0, 1.0] or int, default=1
                When building the vocabulary ignore terms that
                have a document frequency strictly lower than
                the given threshold. This value is also called
                cut-off in the literature. If float, the parameter
                represents a proportion of documents, integer absolute
                counts. This parameter is ignored if vocabulary is not None.
        '''

        # Vectorization using Countvectorize
        cv=CountVectorizer(analyzer=self.tokenize_tweet, max_df=max_df,
                           min_df=min_df, max_features=max_features)
        tweets_bow=cv.fit_transform(tweets)
        feature_names=cv.get_feature_names()
        return tweets_bow, feature_names

    def create_tfidf(self, tweets_bow):
        '''Create the TF-IDF of tweets
        '''
        tfidf_transformer=TfidfTransformer().fit(tweets_bow)
        tweets_tfidf=tfidf_transformer.transform(tweets_bow)
        return tweets_tfidf

    def sentiment_analysis(self, tweet):
        '''Takes a tweet and return a dictionary of scores in 4 categories.
        - Negative score
        - Neutral score
        - Positive score
        - Compound score

        Special characters and stopwords need to stay in the tweet.
        '''

        #create an instance of SentimentIntensityAnalyzer
        sid=SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(tweet)
        if sentiment_scores['compound'] < -0.2:
            sentiment = 'Negative'
        elif -0.2 <= sentiment_scores['compound'] < 0:
            sentiment = 'Neutral\nNegative'
        elif 0 <= sentiment_scores['compound'] <= 0.2:
            sentiment = 'Neutral\nPositive'
        elif sentiment_scores['compound'] > 0.2:
            sentiment = 'Positive'
        else:
            sentiment = None

        return sentiment

    def topic_modeling(self, tweets, num_components=7,
                       num_words_per_topic=10,
                       random_state=42,
                       max_df=1.0, min_df=1,
                       max_features=None):
        '''Get all the tweets and return the topics
        and the highest probability words per topic
        '''

        #create the bags of words
        tweets_bow, feature_names=self.create_bag_of_words(tweets, max_df=max_df,
                                                    min_df=min_df,
                                                    max_features=max_features)

        #create an instace of LatentDirichletAllocation
        #lda=LatentDirichletAllocation(n_components=num_components,
         #                             random_state=random_state)
        #lda.fit(tweets_bow)

        #grab the highest probability words per topic
        ''' words_per_topic={}
        for index, topic in enumerate(lda.components_):
            words_per_topic[index]=[feature_names[i]
                                    for i in topic.argsort()[-15:]]

        topic_results=lda.transform(tweets_bow)
        topics=topic_results.argmax(axis=1)
        return topics, words_per_topic '''

    def find_top_hashtags(self, hashtags):
        '''Get the hashtag column and find the popular hashtags.

        '''
        pass
    
    def sentiment_plot(self, tweets_sentiment):
        '''Create a plot of sentiment analysis.
        '''
        
        logger.info('Creating a plot of sentiment...')
        #fig = plt.figure(figsize=(10, 6))
        img = io.BytesIO()
        sns.countplot(tweets_sentiment, order=['Negative', 'Neutral\nNegative', 'Neutral\nPositive', 'Positive'],
                      palette="RdPu")
        plt.rcParams['font.family'] = "arial"
        plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
        #plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
        plt.xlabel("")
        plt.ylabel("Count", color='silver')
        plt.tick_params(colors='silver')
        plt.tight_layout()
        plt.savefig(img, format='png', facecolor=(
            0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
        img.seek(0)
        figure_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        self.sentiment_plot = 'data:image/png;base64,{}'.format(figure_url)
        return self.sentiment_plot
        
    def run(self, search_query):
        '''
        '''
        
        try:
            #tweetDetective = TweetDetective()
            tweets_list = self.collect_tweets(
                search_query=search_query, geocode="49.525238,-93.874023,4000km")
            df = self.make_dataframe(tweets_list, search_query)
            df['clean_text'] = df['tweet_text'].apply(
                lambda text: self.clean_tweet_text(text, punc_flag=False,
                                                            number_flag=False, special_char_flag=False))
            logger.info('Running sentiment analysis...')
            df['sentiment'] = df['clean_text'].apply(
                lambda tweet: self.sentiment_analysis(tweet))
            sentiment_plot = self.sentiment_plot(df['sentiment'])
            return sentiment_plot

        except Exception as e:
            logger.exception(e)
            return -1




if __name__ == '__main__':
    '''
    '''
    # #logging details
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)
    # #logger = logging.getLogger(__name__)
    # logger = logging.getLogger(sys.argv[0])

    try:
        tweetDetective = TweetDetective()
        tweetDetective.run('Trump')
        # query = input('Enter a search word (Example: Walmart): ')
        # #North America 49.525238,-93.874023,4000km
        # tweets_list = tweetDetective.collect_tweets(
        #     search_query=query, geocode="49.525238,-93.874023,4000km")
        # df = tweetDetective.make_dataframe(tweets_list, query)
        # #df.to_csv('../../processed/tweets.csv')
        # #df = pd.read_csv('../data/processed/tweets.csv')
        # df['clean_text'] = df['tweet_text'].apply(
        #     lambda text: tweetDetective.clean_tweet_text(text, punc_flag=False,
        #                                 number_flag=False, special_char_flag=False))
        # logger.info('Running sentiment analysis...')
        # df['sentiment'] = df['clean_text'].apply(
        #     lambda tweet: tweetDetective.sentiment_analysis(tweet))
        # tweetDetective.sentiment_plot(df['sentiment'])

    except Exception as e:
        logger.exception(e)

    




