
import models

from datetime import datetime, timedelta
import isodate

from peewee import *

import models_trainable

keywod_models = models_trainable.Keyword.select().where(
    models_trainable.Keyword.t_type >= 1,
    models_trainable.Keyword.t_type <= 2
)

key_dict = {}

for keyword_model in keywod_models:
    key_dict[keyword_model.name]= keyword_model

keys = key_dict.keys()


def channels_list_by_username(client, **kwargs):
    kwargs = remove_empty_kwargs(**kwargs)
    response = client.channels().list(
        **kwargs
    ).execute()
    try:
        channelModel = models.Channel.select().where(models.Channel.name ** kwargs['forUsername']).get()
    except models.DoesNotExist:
        channelModel = models.Channel.create(name=kwargs['forUsername'])

    for channel in response["items"]:
        # From the API response, extract the playlist ID that identifies the list
        # of videos uploaded to the authenticated user's channel.
        uploads_list_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]

        print("Videos in list %s" % uploads_list_id)

        playlists_list_by_id(
            client,
            channelModel,
            playlistId=uploads_list_id,
            part="snippet, contentDetails",
            maxResults=50)


def channels_playlist_by_username(client, **kwargs):
    kwargs = remove_empty_kwargs(**kwargs)
    response = client.channels().list(
        **kwargs
    ).execute()
    try:
        channelModel = models.Channel.select().where(models.Channel.name ** kwargs['forUsername']).get()
    except models.DoesNotExist:
        channelModel = models.Channel.create(name=kwargs['forUsername'])

    for channel in response["items"]:
        # From the API response, extract the playlist ID that identifies the list
        # of videos uploaded to the authenticated user's channel.
        uploads_list_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]

        print("Videos in list %s" % uploads_list_id)

        playlists_list_by_id(
            client,
            channelModel,
            playlistId=uploads_list_id,
            part="snippet, contentDetails",
            maxResults=50)


def playlists_list_by_id(client, channelModel, **kwargs):
    playlistitems_list_request = client.playlistItems().list(
        **kwargs
    )

    while playlistitems_list_request:
        playlistitems_list_response = playlistitems_list_request.execute()

        publishedAt = ""

        # Print information about each video.
        for playlist_item in playlistitems_list_response["items"]:
            title = playlist_item["snippet"]["title"]
            video_id = playlist_item["snippet"]["resourceId"]["videoId"]
            content_details = playlist_item["snippet"]["description"]
            publishedAt = playlist_item["snippet"]["publishedAt"]

            video_response = videos_list_by_id(client,
                              part='snippet,contentDetails',
                              id=video_id)
            duration = video_response["items"][0]["contentDetails"]["duration"]

            try:
                pubTime = datetime.strptime(publishedAt[:19], "%Y-%m-%dT%H:%M:%S")
                durTime = isodate.parse_duration(duration).total_seconds()
                if pubTime+ timedelta(days=1) > datetime.utcnow():
                    models.Video.create(
                        publishedAt=pubTime,
                        title=title,
                        content=content_details,
                        channel=channelModel,
                        videoId=video_id,
                        duration=durTime
                    )

            except IntegrityError:
                print("Already Exist : {}".format(title))
                pass

        if pubTime+timedelta(days=1) > datetime.utcnow():
            print("Ok : ",datetime.strptime(publishedAt[:19], "%Y-%m-%dT%H:%M:%S"), datetime.utcnow())
            playlistitems_list_request = client.playlistItems().list_next(
                playlistitems_list_request, playlistitems_list_response)
        else:
            print("Done : ", datetime.strptime(publishedAt[:19], "%Y-%m-%dT%H:%M:%S"), datetime.utcnow())
            break


def playlists_list_by_id_endless(client, channelModel, **kwargs):
    playlistitems_list_request = client.playlistItems().list(
        **kwargs
    )

    while playlistitems_list_request:
        playlistitems_list_response = playlistitems_list_request.execute()

        publishedAt = ""

        # Print information about each video.
        for playlist_item in playlistitems_list_response["items"]:
            title = playlist_item["snippet"]["title"]
            video_id = playlist_item["snippet"]["resourceId"]["videoId"]
            content_details = playlist_item["snippet"]["description"]
            publishedAt = playlist_item["snippet"]["publishedAt"]

            video_response = videos_list_by_id(client,
                              part='snippet,contentDetails',
                              id=video_id)
            duration = video_response["items"][0]["contentDetails"]["duration"]

            video_model = None

            try:
                pubTime = datetime.strptime(publishedAt[:19], "%Y-%m-%dT%H:%M:%S")

                video_model = models_trainable.Video.create(
                    publishedAt=pubTime,
                    title=title,
                    content=content_details,
                    originalLink=video_id
                )
                print("CR: {}".format(title))
            except IntegrityError:
                pass
            """
            try:

                if not video_model:
                    video_model = models_trainable.Video.get(
                        models_trainable.Video.originalLink == video_id)

            except DoesNotExist:
                pass

            if video_model:
                for key in keys:
                    if key in title:
                        try:
                            models_trainable.Relationship.create(
                                from_keyword=key_dict[key],
                                to_video=video_model
                            )
                            print("SR: {} --- {}".format(key, title))

                        except IntegrityError:
                            pass
            """

        print("Ok : ", datetime.strptime(publishedAt[:19], "%Y-%m-%dT%H:%M:%S"), datetime.utcnow())
        playlistitems_list_request = client.playlistItems().list_next(
            playlistitems_list_request, playlistitems_list_response)

def videos_list_by_id(client, **kwargs):
  # See full sample for function
  kwargs = remove_empty_kwargs(**kwargs)

  response = client.videos().list(
    **kwargs
  ).execute()

  return response


def remove_empty_kwargs(**kwargs):
    good_kwargs = {}
    if kwargs is not None:
        for key, value in kwargs.items():
            if value:
                good_kwargs[key] = value
    return good_kwargs
