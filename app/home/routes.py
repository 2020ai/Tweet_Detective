# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app.home import blueprint
from flask import render_template, redirect, url_for, request
from app import login_manager
from jinja2 import TemplateNotFound
from app.home import TweetDetective


@blueprint.route('/')
def route_default():
    return redirect(url_for('home_blueprint.search'))

@blueprint.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        if request.form['search']:
            search_query = request.form['search']
            print('search term: ', search_query)
            tweetDetective = TweetDetective.TweetDetective()
            sentiment_plot = tweetDetective.run(search_query)
            return render_template('index.html', segment='index', sentiment_plot=sentiment_plot)
            #return redirect(url_for('home_blueprint.index'))
        else:
            return render_template('login.html')

@blueprint.route('/index')
def index():

    return render_template('index.html', segment='index')

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
