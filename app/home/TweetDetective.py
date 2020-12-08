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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


# Twitter API credentials
ACCESS_TOKEN = ["761441357315440640-suCCQJo6kuufi3PmcYUl2y9kNyYb8C0",
                "787838102219984896-FM1snnheZBR9bDKHp8WfQQ8LOPHzBQO"]
ACCESS_TOKEN_SECRET = ["nN4nX0LhlUZHN31LLYU1neOxg7elvb4LIo9KkX7gMDMaN",
                       "CRJnu5dNbScNXs3ZS1wti63UZWMZN3TNVllp6rjVs5jXu"]
API_KEY = ["oMlZlYVi6MerYj7SZzcYWvgVr", "RVgFvwUTzge2JUK6yuav4ZYoM"]
API_SECRET_KEY = ["OW8cYRS69LUQ1gD5rKULGi4QtuBoj0OX5hRyJI5HVBbzTLZzam",
                  "XJaKlhIw5sTGxlgDnftJ6x7eOh81203QVRpQkjhqtbrhVHcBIx"]

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
                       num_of_page=20, count=100, since=None, until=None):
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
        api_key = API_KEY[0]
        api_secret_key = API_SECRET_KEY[0]
        access_token = ACCESS_TOKEN[0]
        access_token_secret = ACCESS_TOKEN_SECRET[0]
        access_keys_flag = True #The ability to use 2 set of authentication

        if search_query=='':
            tweets_list = []
        else:
            rate_limit = 1  # There is a limit of 100 API calls in the hour
            if rate_limit < 1:
                    # Rate limit time out needs to be added here in order to
                    # collect data exceeding available rate-limit
                    if access_keys_flag:
                        api_key = API_KEY[1]
                        api_secret_key = API_SECRET_KEY[1]
                        access_token = ACCESS_TOKEN[1]
                        access_token_secret = ACCESS_TOKEN_SECRET[1]
                        access_keys_flag = False
                    else:
                        logger.exception('{} Rate limit!'.format(rate_limit))
            try:
                
                twitter_obj = Twython(api_key, api_secret_key,
                                      access_token, access_token_secret)

                # Use Twitter standard API search
                tweet_result = twitter_obj.search(q=search_query, geocode=geocode,
                                                result_type=result_type, count=count,
                                                since=since, until=until,
                                                include_entities='true',
                                                tweet_mode='extended', lang='en')
            except Exception as e:
                logger.exception(e)
                return -1
            rate_limit = int(twitter_obj.get_lastfunction_header(
                'x-rate-limit-remaining'))

            # In order to prevent redundant tweets explained here
            # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
            # instead of reading a timeline relative to the top of the list
            # (which changes frequently), an application should read the timeline
            # relative to the IDs of tweets it has already processed.
            tweets_list = tweet_result['statuses']
            i = 0  # num of iteration through each page
            #rate_limit = 1  # There is a limit of 100 API calls in the hour
            while tweet_result['statuses'] and i < num_of_page:
                if rate_limit < 1:
                    # Rate limit time out needs to be added here in order to
                    # collect data exceeding available rate-limit
                    if access_keys_flag:
                        api_key = API_KEY[1]
                        api_secret_key = API_SECRET_KEY[1]
                        access_token = ACCESS_TOKEN[1]
                        access_token_secret = ACCESS_TOKEN_SECRET[1]
                        access_keys_flag = False
                    else:
                        logger.exception('{} Rate limit!'.format(rate_limit))
                        break
                max_id = tweet_result['statuses'][-1]['id']-1

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

        self.num_of_tweets = len(tweets_list)
        return tweets_list

    def find_hashtags(self, tweet):
        '''Find the hashtags in a tweet.
        '''

        hashtags = ''
        for term in tweet:
            hashtags += term['text']+','
        return hashtags

    def make_dataframe(self, tweets_list, search_term):
        '''Gets the list of tweets and return it as a pandas DataFrame.
        '''

        logger.info('Creating dataframe from tweets...')
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
            map(lambda tweet: self.find_hashtags(tweet['entities']['hashtags']) if tweet['entities']['hashtags'] else None, tweets_list))
        self.tweets_df['search_term'] = list(map(lambda tweet: search_term if search_term.lower(
        ) in tweet['full_text'].lower() else None, tweets_list))

        #remove tweets without search term
        self.tweets_df.dropna(subset=['search_term'], inplace=True)
        #drop duplicates
        logger.info('Removing duplicate tweets')
        self.tweets_df.drop_duplicates(
            subset='tweet_text', inplace=True, ignore_index=True)
        

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
            tweet = re.sub(r'@[w\w]+', ' ', tweet)

        # remove the urls
        if urls_flag:
            tweet = re.sub(
                r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', tweet)

        # replace the negations
        tweet = re.sub(r"n't", ' not', tweet)
        tweet = re.sub(r"N'T", ' NOT', tweet)

        # remove punctuations
        if punc_flag:
            tweet = re.sub(
                '[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’…”'']+', ' ', tweet)

        # remove numbers
        if number_flag:
            tweet = re.sub('([0-9]+)', '', tweet)

        # remove special characters
        if special_char_flag:
            tweet = re.sub(r'[^\w]', ' ', tweet)

        # remove double space
        tweet = re.sub('\s+', ' ', tweet)

        if stop_word_flag:
            tweet = ' '.join([word for word in tweet.split() if word.lower(
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
        cv = CountVectorizer(analyzer=self.tokenize_tweet, max_df=max_df,
                             min_df=min_df, max_features=max_features, 
                             stop_words = stopwords.words('english')+STOP_WORDS)
        tweets_bow = cv.fit_transform(tweets)
        cv_feature_names = cv.get_feature_names()
        return tweets_bow, cv_feature_names

    def create_tfidf(self, tweets, max_df=0.95, min_df=2,
                            max_features=None):
        '''Create the TF-IDF of tweets
        '''
        tfidf = TfidfVectorizer(analyzer=self.tokenize_tweet, max_df=max_df,
                             min_df=min_df, max_features=max_features,
                             stop_words = stopwords.words('english')+STOP_WORDS)
        tweets_tfidf = tfidf.fit_transform(tweets)
        tfidf_feature_names = tfidf.get_feature_names()
        return tweets_tfidf, tfidf_feature_names

    def sentiment_analysis(self, tweet):
        '''Takes a tweet and return the sentiment scores which is
        between -1 and 1.
        
        Special characters and stopwords need to stay in the tweet.
        '''

        #create an instance of SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(tweet)
        return sentiment_scores['compound']

    def sentiment_analysis_category(self, sentiment_score):
        '''Takes a tweet's sentimetn score and return one of 
        the 4 following categories.
        - Negative 
        - Neutral 
        - Positive 
        - Compound 

        Special characters and stopwords need to stay in the tweet.
        '''

        if sentiment_score < -0.2:
            self.sentiment = 'Negative'
        elif -0.2 <= sentiment_score < 0:
            self.sentiment = 'Neutral\nNegative'
        elif 0 <= sentiment_score <= 0.2:
            self.sentiment = 'Neutral\nPositive'
        elif sentiment_score > 0.2:
            self.sentiment = 'Positive'
        else:
            self.sentiment = None

        return self.sentiment
        

    def topic_modeling(self, tweets, num_components=4,
                       num_words_per_topic=20,
                       random_state=42,
                       max_df=0.95, min_df=2,
                       max_features=None):
        '''Get all the tweets and return the topics
        and the highest probability words per topic
        '''

        words_per_topic = {}
        if len(tweets) > 0:
            #create the bags of words
            tweets_bow, cv_feature_names = self.create_bag_of_words(tweets, max_df=max_df,
                                                                min_df=min_df,
                                                                max_features=max_features)

            #create an instace of LatentDirichletAllocation
            LDA=LatentDirichletAllocation(n_components=num_components, random_state=random_state)
            LDA.fit(tweets_bow)

            #grab the highest probability words per topic
            
            for index, topic in enumerate(LDA.components_):
                words_per_topic[index]=[cv_feature_names[i]
                                        for i in topic.argsort()[-20:]]
            topic_results = LDA.transform(tweets_bow)
            topics = topic_results.argmax(axis=1)
        else:
            topics = []

        return topics, words_per_topic

    def plot_topic_wordcloud(self, tweets_df):
        '''Plots the WordCloud for each topic
        '''
        if len(tweets_df) > 0:
            mask = np.array(Image.open("wordcloud_template_square.jpg"))
            wc = WordCloud(font_path='CabinSketch-Bold.ttf', background_color='black', 
                        mask=mask, mode='RGB',
                        width=1000, max_words=50, height=1000, relative_scaling=0.5,
                        random_state=1, contour_width=10, contour_color='white', 
                        stopwords=stopwords.words('english')+STOP_WORDS)
            self.topic_wordcloud_pics = []
            for i in range(4):
                img = io.BytesIO()
                text_all = ''
                text_all = ' '.join(text for text in tweets_df[tweets_df['topic']==i]['clean_text'])
                wc.generate(text_all)
                plt.figure(figsize=(10, 10))
                plt.imshow(wc, interpolation='bilinear')
                plt.tight_layout(pad=0)
                plt.axis('off')
                plt.savefig(img, format='png', facecolor=(
                    0, 0, 0, 1), edgecolor=(0, 0, 0, 1))
                img.seek(0)
                figure_url = base64.b64encode(img.getvalue()).decode()
                plt.close()
                self.topic_wordcloud_pics.append('data:image/png;base64,{}'.format(figure_url))
        else:
            img = io.BytesIO()
            plt.figure(figsize=(10, 10))
            plt.text(0.5, 0.5, "Not Enough Tweets Found \n for the Past Few Days", size=40,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),))
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.axis('off')
            plt.tick_params(colors='silver')
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            topic_wordcloud_pic = 'data:image/png;base64,{}'.format(figure_url)
            self.topic_wordcloud_pics = [topic_wordcloud_pic]*4
        return self.topic_wordcloud_pics

    def plot_topic_count(self, topics):
        '''countplot for topics
        '''
        logger.info('Creating a plot of all topic counts...')
        
        if len(topics) > 0:
            topic_dict = {0:'Topic 1', 1:'Topic 2', 2:'Topic 3', 3:'Topic 4'}
            topics = topics.map(topic_dict)
            img = io.BytesIO()
            plt.figure(figsize=(6, 3))
            sns.countplot(topics, palette="RdPu", order=[
                          'Topic 1', 'Topic 2',  'Topic 3', 'Topic 4'])
            plt.rcParams['font.family'] = "arial"
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.tick_params(colors='silver')
            plt.xlabel("")
            plt.ylabel("Count", color='silver')        
            plt.tight_layout()
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.topic_count_plot = 'data:image/png;base64,{}'.format(figure_url)
        else:
            img = io.BytesIO()
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "Not Enough Tweets Found \nfor the Past Few Days", size=20,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),))
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.axis('off')
            plt.tick_params(colors='silver')
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.topic_count_plot = 'data:image/png;base64,{}'.format(figure_url)

        return self.topic_count_plot

    
    def plot_popular_hashtags(self, hashtags, max_hash=10):
        '''Get the hashtag column and plot the popular hashtags
        '''

        logger.info('Creating a plot of popular hashtags...')
        hashtags.dropna(inplace=True)
        hashtags_list = []
        for hashtag in hashtags:
            hashtags_list += hashtag[:-1].lower().split(',')
        hashtags_series = pd.Series(hashtags_list)
        
        if hashtags_series.nunique() < max_hash :
            max_hash = hashtags_series.nunique()
        if max_hash != 0 :
            img = io.BytesIO()
            plt.figure(figsize=(10, 4))
            sns.countplot(hashtags_series, order=hashtags_series.value_counts().iloc[:max_hash].index)
            plt.rcParams['font.family'] = "arial"
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.xlabel("")
            plt.ylabel("Count", color='silver')
            plt.xticks(rotation=15, ha='right')
            plt.tick_params(colors='silver')
            plt.tight_layout()
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.hashtag_plot = 'data:image/png;base64,{}'.format(figure_url)
        else:
            img = io.BytesIO()
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, "Not Enough Tweets Found \nfor the Past Few Days", size=25,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                            ec=(1., 0.5, 0.5),
                            fc=(1., 0.8, 0.8),))
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.axis('off')
            plt.tick_params(colors='silver')
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.hashtag_plot = 'data:image/png;base64,{}'.format(figure_url)
        
        return self.hashtag_plot

    def plot_sentiment_analysis(self, tweets_sentiment):
        '''Create a plot of sentiment analysis.
        '''

        logger.info('Creating a plot of sentiment...')
        
        if len(tweets_sentiment) > 0:
            img = io.BytesIO()
            plt.figure(figsize=(6, 3))
            sns.countplot(tweets_sentiment, order=['Negative', 'Neutral\nNegative', 'Neutral\nPositive', 'Positive'],
                        palette="RdPu")
            plt.rcParams['font.family'] = "arial"
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.tick_params(colors='silver')
            plt.xlabel("")
            plt.ylabel("Count", color='silver')         
            plt.tight_layout()
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.sentiment_plot = 'data:image/png;base64,{}'.format(figure_url)
        else:
            img = io.BytesIO()
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "Not Enough Tweets Found \nfor the Past Few Days", size=20,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8),))
            plt.rcParams['axes.facecolor'] = (0.22, 0.23, 0.31, 1)
            plt.rcParams['axes.edgecolor'] = (0.22, 0.23, 0.31, 1)
            plt.axis('off')
            plt.tick_params(colors='silver')
            plt.savefig(img, format='png', facecolor=(
                0.22, 0.23, 0.31, 1), edgecolor=(0.22, 0.23, 0.31, 1))
            img.seek(0)
            figure_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            self.sentiment_plot = 'data:image/png;base64,{}'.format(figure_url)

        return self.sentiment_plot

    def find_top_pos_neg_tweets(self, top_num=5):
        '''Create a series of top negative and positive tweets.

        The tweets can be accessed through iloc[]
        '''
        logger.info('Finding the top pos and neg tweets...')
        if len(self.tweets_df['sentiment_score']) < top_num:
            self.top_negative_tweets = pd.Series(
                ['Not Enough Tweets Found for the Past Few Days']*5)
            self.top_positive_tweets = pd.Series(
                ['Not Enough Tweets Found for the Past Few Days']*5)
        else:
            self.top_negative_tweets = self.tweets_df['tweet_text'][self.tweets_df['sentiment_score'].argsort()[
                :top_num]]
            self.top_positive_tweets = self.tweets_df['tweet_text'][self.tweets_df['sentiment_score'].argsort(
            )[-top_num:]]
        return self.top_negative_tweets, self.top_positive_tweets

    def run(self, search_query):
        '''
        '''

        try:
            self.search_query = search_query
            tweets_list = self.collect_tweets(
                search_query=search_query, geocode="49.525238,-93.874023,4000km")
            self.tweets_df = self.make_dataframe(tweets_list, search_query)
            self.tweets_df['clean_text'] = self.tweets_df['tweet_text'].apply(
                lambda text: self.clean_tweet_text(text, punc_flag=False,
                                                   number_flag=False, special_char_flag=False))
            logger.info('Running sentiment analysis...')
            self.tweets_df['sentiment_score'] = self.tweets_df['clean_text'].apply(
                lambda tweet: self.sentiment_analysis(tweet))
            self.tweets_df['sentiment'] = self.tweets_df['sentiment_score'].apply(
                lambda sentiment_score: self.sentiment_analysis_category(sentiment_score))
            self.plot_sentiment_analysis(self.tweets_df['sentiment'])
            self.plot_sentiment_analysis(self.tweets_df['sentiment'])
            self.plot_popular_hashtags(self.tweets_df['hashtags'])
            self.find_top_pos_neg_tweets()
            
            logger.info('Running topic modeling...')
            self.tweets_df['topic'], self.words_per_topic = self.topic_modeling(self.tweets_df['clean_text'])
            self.plot_topic_wordcloud(self.tweets_df)
            self.plot_topic_count(self.tweets_df['topic'])
            return 0

        except Exception as e:
            logger.exception(e)
            return -1



if __name__ == '__main__':
    '''
    '''

    try:
        search_query = input('Enter your search term: ')
        tweetDetective = TweetDetective()
        tweetDetective.run(search_query)

    except Exception as e:
        logger.exception(e)

