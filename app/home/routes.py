# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for, request
from jinja2 import TemplateNotFound
from app.home import TweetDetective


@blueprint.route('/')
def route_default():
    return redirect(url_for('home_blueprint.search'))

@blueprint.route('/search', methods=['GET', 'POST'])
def search():

    global SEARCH_QUERY1, SEARCH_QUERY2, TWEETDETECTIVE1, TWEETDETECTIVE2

    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        if request.form['search1'] and request.form['search2']:
            SEARCH_QUERY1 = request.form['search1']
            print('search term: ', SEARCH_QUERY1)
            TWEETDETECTIVE1 = TweetDetective.TweetDetective()
            TWEETDETECTIVE1.run(SEARCH_QUERY1)
            SEARCH_QUERY2 = request.form['search2']
            print('search term: ', SEARCH_QUERY2)
            TWEETDETECTIVE2 = TweetDetective.TweetDetective()
            TWEETDETECTIVE2.run(SEARCH_QUERY2)
            return redirect(url_for('home_blueprint.index'))
        elif request.form['search1'] and not request.form['search2']:
            SEARCH_QUERY1 = request.form['search1']
            print('search term: ', SEARCH_QUERY1)
            TWEETDETECTIVE1 = TweetDetective.TweetDetective()
            TWEETDETECTIVE1.run(SEARCH_QUERY1)
            SEARCH_QUERY2 = ''
            print('search term: ', SEARCH_QUERY2)
            TWEETDETECTIVE2 = TweetDetective.TweetDetective()
            TWEETDETECTIVE2.run(SEARCH_QUERY2)
            return redirect(url_for('home_blueprint.index'))
        elif request.form['search2'] and not request.form['search1']:
            SEARCH_QUERY1 = ''
            print('search term: ', SEARCH_QUERY1)
            TWEETDETECTIVE1 = TweetDetective.TweetDetective()
            TWEETDETECTIVE1.run(SEARCH_QUERY1)
            SEARCH_QUERY2 = request.form['search2']
            print('search term: ', SEARCH_QUERY2)
            TWEETDETECTIVE2 = TweetDetective.TweetDetective()
            TWEETDETECTIVE2.run(SEARCH_QUERY2)
            return redirect(url_for('home_blueprint.index'))
        else:
            return render_template('login.html')


@blueprint.route('/index')
def index():

    #return render_template('index.html', segment='index')
    return render_template('index.html', segment='index', 
                           search_query1=TWEETDETECTIVE1.search_query,
                           sentiment_plot1=TWEETDETECTIVE1.sentiment_plot,
                           num_of_tweets1=TWEETDETECTIVE1.num_of_tweets,
                           hashtag_plot1=TWEETDETECTIVE1.hashtag_plot,
                           top_positive_tweets1_1=TWEETDETECTIVE1.top_positive_tweets.iloc[4],
                           top_positive_tweets1_2=TWEETDETECTIVE1.top_positive_tweets.iloc[3],
                           top_positive_tweets1_3=TWEETDETECTIVE1.top_positive_tweets.iloc[2],
                           top_positive_tweets1_4=TWEETDETECTIVE1.top_positive_tweets.iloc[1],
                           top_positive_tweets1_5=TWEETDETECTIVE1.top_positive_tweets.iloc[0],
                           top_negative_tweets1_1=TWEETDETECTIVE1.top_negative_tweets.iloc[0],
                           top_negative_tweets1_2=TWEETDETECTIVE1.top_negative_tweets.iloc[1],
                           top_negative_tweets1_3=TWEETDETECTIVE1.top_negative_tweets.iloc[2],
                           top_negative_tweets1_4=TWEETDETECTIVE1.top_negative_tweets.iloc[3],
                           top_negative_tweets1_5=TWEETDETECTIVE1.top_negative_tweets.iloc[4],
                           topic_wordcloud_1_1=TWEETDETECTIVE1.topic_wordcloud_pics[0],
                           topic_wordcloud_1_2=TWEETDETECTIVE1.topic_wordcloud_pics[1],
                           topic_wordcloud_1_3=TWEETDETECTIVE1.topic_wordcloud_pics[2],
                           topic_wordcloud_1_4=TWEETDETECTIVE1.topic_wordcloud_pics[3],
                           topic_count_1=TWEETDETECTIVE1.topic_count_plot,
                           search_query2=TWEETDETECTIVE2.search_query,
                           sentiment_plot2=TWEETDETECTIVE2.sentiment_plot,
                           num_of_tweets2=TWEETDETECTIVE2.num_of_tweets,
                           hashtag_plot2=TWEETDETECTIVE2.hashtag_plot,
                           top_positive_tweets2_1=TWEETDETECTIVE2.top_positive_tweets.iloc[4],
                           top_positive_tweets2_2=TWEETDETECTIVE2.top_positive_tweets.iloc[3],
                           top_positive_tweets2_3=TWEETDETECTIVE2.top_positive_tweets.iloc[2],
                           top_positive_tweets2_4=TWEETDETECTIVE2.top_positive_tweets.iloc[1],
                           top_positive_tweets2_5=TWEETDETECTIVE2.top_positive_tweets.iloc[0],
                           top_negative_tweets2_1=TWEETDETECTIVE2.top_negative_tweets.iloc[0],
                           top_negative_tweets2_2=TWEETDETECTIVE2.top_negative_tweets.iloc[1],
                           top_negative_tweets2_3=TWEETDETECTIVE2.top_negative_tweets.iloc[2],
                           top_negative_tweets2_4=TWEETDETECTIVE2.top_negative_tweets.iloc[3],
                           top_negative_tweets2_5=TWEETDETECTIVE2.top_negative_tweets.iloc[4],
                           topic_wordcloud_2_1=TWEETDETECTIVE2.topic_wordcloud_pics[0],
                           topic_wordcloud_2_2=TWEETDETECTIVE2.topic_wordcloud_pics[1],
                           topic_wordcloud_2_3=TWEETDETECTIVE2.topic_wordcloud_pics[2],
                           topic_wordcloud_2_4=TWEETDETECTIVE2.topic_wordcloud_pics[3],
                           topic_count_2=TWEETDETECTIVE2.topic_count_plot)

@blueprint.route('/<template>')
def route_template(template):

    try:

        if not template.endswith( '.html' ):
            template += '.html'

        # Detect the current page
        segment = get_segment( request )

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( template, segment=segment )

    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500

# Helper - Extract current page name from request 
def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment    

    except:
        return None  
