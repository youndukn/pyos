from peewee import *
import datetime

DATABASE = SqliteDatabase('news_restful.db')


class Daily(Model):
    timestamp = DateTimeField(default=datetime.datetime.now)

    class Meta:
        database = DATABASE
        order_by = ('-timestamp', )

class Channel(Model):
    name = TextField()

    class Meta:
        database = DATABASE
        order_by = ('-timestamp', )


class Video(Model):
    timestamp = DateTimeField(default=datetime.datetime.now)
    publishedAt = DateTimeField()
    title = TextField()
    content = TextField()
    videoId = CharField(unique=True)
    duration = IntegerField()

    channel = ForeignKeyField(
        model=Channel,
        related_name='channel'
    )

    class Meta:
        database = DATABASE
        order_by = ('-timestamp', )



def initialized():
    DATABASE.connect()
    DATABASE.create_tables((Daily, Channel, Video), safe=True)
    DATABASE.close()