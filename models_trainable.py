from peewee import *
import datetime

DATABASE = SqliteDatabase('news_trainable.db')


class Keyword(Model):
    name = CharField(unique=True)
    t_type = IntegerField()

    class Meta:
        database = DATABASE
        order_by = ('-timestamp', )


class Video(Model):
    timestamp = DateTimeField(default=datetime.datetime.now)
    publishedAt = TimeField()
    title = TextField()
    content = TextField()
    originalLink = CharField(unique=True)

    class Meta:
        database = DATABASE
        order_by = ('-timestamp', )


class Relationship(Model):
    from_keyword = ForeignKeyField(Keyword, related_name='relationships')
    to_video = ForeignKeyField(Video, related_name='video_to')

    class Meta:
        database = DATABASE
        indexes = (
            (('from_keyword', 'to_video'), True),
        )



def initialized():
    DATABASE.connect()
    DATABASE.create_tables((Keyword, Video, Relationship), safe=True)
    DATABASE.close()