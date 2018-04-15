# -*- coding: utf-8 -*-

import os

import flask
from flask import render_template, redirect, url_for

import google.oauth2.credentials
import google_auth_oauthlib.flow
import googleapiclient.discovery

import models

from datetime import datetime, timedelta

import forms

import youtube
import naver

from process_dailies import Dailies

news_list = ['tvchosun01', 'sbsnews8', 'JTBC10news', 'NewsKBS', 'MBCnews']

block_list = []

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

app = flask.Flask(__name__)
# Note: A secret key is included in the sample so that it works, but if youp
# use this code in your application please replace this with a truly secret
# key. See http://flask.pocoo.org/docs/0.12/quickstart/#sessions.
app.secret_key = 'REPLACE ME - this value is here as a placeholder.'


@app.route('/')
def index():

    channels = [[], [], [], [], []]
    for i, news in enumerate(news_list):
        channel = models.Channel.select().where(models.Channel.name ** news).get()
        videos_c = models.Video.select().order_by(models.Video.publishedAt.desc()).where(
            models.Video.channel == channel,
            models.Video.publishedAt > datetime.utcnow() - timedelta(days=1)
        )
        channels[i] = videos_c

    dailies = Dailies(channels)

    if dailies:
        return render_template('new_video_stream.html', stream=dailies.process_vector_relevance())
    else:
        return render_template('layout.html')


@app.route('/admin/<type>', methods=('GET', 'POST'))
@app.route('/admin', methods=('GET', 'POST'))
def admin(type=None):
    form = forms.AdminForm()
    if form.validate_on_submit():
        if type == "youtube" or form.stream_type.data.lower() == "youtube":

            if 'credentials' not in flask.session:
                return flask.redirect('authorize')

            # Load the credentials from the session.
            credentials = google.oauth2.credentials.Credentials(
                **flask.session['credentials'])

            client = googleapiclient.discovery.build(
                API_SERVICE_NAME, API_VERSION, credentials=credentials)

            for i, news in enumerate(news_list):
                if not (i in block_list):
                    print(news)
                    youtube.channels_list_by_username(client,
                                              part='snippet,contentDetails',
                                              forUsername=news)
        elif type == "naver" or form.stream_type.data.lower() == "naver":
            naver.crawl_naver()

        return redirect(url_for('index'))

    return render_template('admin.html', form=form)


@app.route('/authorize')
def authorize():
    # Create a flow instance to manage the OAuth 2.0 Authorization Grant Flow
    # steps.
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)
    authorization_url, state = flow.authorization_url(
        # This parameter enables offline access which gives your application
        # both an access and refresh token.
        access_type='offline',
        # This parameter enables incremental auth.
        include_granted_scopes='true')

    # Store the state in the session so that the callback can verify that
    # the authorization server response.
    flask.session['state'] = state

    return flask.redirect(authorization_url)


@app.route('/oauth2callback')
def oauth2callback():
    # Specify the state when creating the flow in the callback so that it can
    # verify the authorization server response.
    state = flask.session['state']
    flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = flask.url_for('oauth2callback', _external=True)

    # Use the authorization server's response to fetch the OAuth 2.0 tokens.
    authorization_response = flask.request.url
    flow.fetch_token(authorization_response=authorization_response)

    # Store the credentials in the session.
    # ACTION ITEM for developers:
    #     Store user's access and refresh tokens in your data store if
    #     incorporating this code into your real app.
    credentials = flow.credentials
    flask.session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

    return flask.redirect(flask.url_for('index'))


if __name__ == '__main__':
    models.initialized()

    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run('localhost', 8090, debug=True)
