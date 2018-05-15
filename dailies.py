from flask import Blueprint, abort

from flask_restful import (Resource, Api, reqparse,
                           inputs, fields, marshal, marshal_with, url_for)
import array
import models
import models_trainable
import json
import numpy

from datetime import datetime, timedelta

news_list = ['JTBC10news', 'MBCnews', 'sbsnews8', 'tvchosun01']

video_fields = {
    'id' : fields.Integer,
    'videoId' : fields.String,
    'title': fields.String,
    'vector' : fields.Raw
}


def videos_or_404():
    try:
        videos = []
        for i, news in enumerate(news_list):
            channel = models.Channel.select().where(models.Channel.name ** news).get()
            videos_c = models.Video.select().order_by(models.Video.publishedAt.desc()).where(
                models.Video.channel == channel,
                models.Video.publishedAt > datetime.utcnow() - timedelta(days=1)
            )
            for video in videos_c:
                videos.append(video)

    except models.Video.DoesNotExist:
        abort(404)
    else:
        return videos

def keywords_or_404():
    try:
        keyword_models = models_trainable.Keyword.select().where(
            models_trainable.Keyword.t_type >= 1,
            models_trainable.Keyword.t_type <= 2
        )

    except models.Video.DoesNotExist:
        abort(404)
    else:
        return keyword_models



class Video(Resource):

    def get(self):

        videos = []

        for video in videos_or_404():
            video_field = marshal(video, video_fields)
            video_field["channel"] = video.channel.name
            video_field["vector"] = numpy.frombuffer(video_field["vector"]).tolist()
            videos.append(video_field)

        return {'videos': videos}


dailies_api = Blueprint('resources.dailies', __name__)
api = Api(dailies_api)

api.add_resource(
    Video,
    '/api/v1/videos/',
    endpoint='videos'
)