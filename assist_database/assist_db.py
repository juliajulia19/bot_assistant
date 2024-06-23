from peewee import SqliteDatabase, Model

database = SqliteDatabase('assist_db.sqlite')


class BaseModel(Model):
    class Meta:
        database = database