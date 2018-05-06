from flask import Blueprint, abort

from flask_restful import (Resource, Api, reqparse,
                           inputs, fields, marshal, marshal_with, url_for)

import models
from datetime import datetime, timedelta

news_list = ['JTBC10news', 'MBCnews', 'sbsnews8', 'tvchosun01']

video_fields = {
    'id' : fields.Integer,
    'videoId' : fields.String,
    'title': fields.String,
    'vector' : fields.String
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


class Video(Resource):

    def get(self):
        videos = [marshal(video, video_fields) for video in videos_or_404()]
        return {'videos': videos}


dailies_api = Blueprint('resources.dailies', __name__)
api = Api(dailies_api)

api.add_resource(
    Video,
    '/api/v1/videos/',
    endpoint='videos'
)