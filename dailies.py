from flask import Blueprint, abort

from flask_restful import (Resource, Api, reqparse,
                           inputs, fields, marshal, marshal_with, url_for)

import models_restful as models

daily_fields = {
    'timestamp' : fields.DateTime,
    'id' : fields.String
}

video_fields = {
    'videoId' : fields.Integer,
    'duration' : fields.Integer,
    'title': fields.String,
    'content' : fields.String
}


def daily_or_404(daily_id):
    try:
        daily = models.Daily.get(models.Daily.id==daily_id)
    except models.Daily.DoesNotExist:
        abort(404)
    else:
        return daily


def video_or_404(video_id):
    try:
        video = models.Video.get(models.Daily.id==video_id)
    except models.Video.DoesNotExist:
        abort(404)
    else:
        return video


class DailyList(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            'title',
            require=True,
            help='No course title provided',
            location=['form', 'json']
        )
        self.reqparse.add_argument(
            'url',
            require=True,
            help='No course URL provided',
            location=['form', 'json'],
            type=inputs.url
        )
        super().__init__()

    def get(self):
        dailies = [marshal(daily, daily_fields)
                   for daily in models.Daily.select()]
        return {'dailies': dailies}


class Daily(Resource):
    def get(self, id):
        video = [marshal(video, daily_fields)
                   for video in models.Video.select()]
        return {'dailies': video}

class Video(Resource):

    @marshal_with(video_fields)
    def get(self, id):
        return daily_or_404(id)


dailies_api = Blueprint('resources.dailies', __name__)
api = Api(dailies_api)
api.add_resource(
    DailyList,
    '/api/v1/dailies',
    endpoin='dailies'
)

api.add_resource(
    Daily,
    '/api/v1/dailies/<int:id>',
    endpoin='daily'
)

api.add_resource(
    Video,
    '/api/v1/dailies/<int:id>',
    endpoin='video'
)