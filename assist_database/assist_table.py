from peewee import PrimaryKeyField, TextField, IntegerField

from assist_database.assist_db import BaseModel

class Procurement(BaseModel):
    user_id = PrimaryKeyField(null=False)
    customer = TextField(null=False)
    telegram_id = IntegerField()
    product = TextField()
    quantity = TextField()



    @staticmethod
    def from_telegram_id(tg_id):
        return Procurement.get(Procurement.telegram_id == tg_id)

    @staticmethod
    def delete_user_by_telegram_id(tg_id):
        Appointments.delete().where(Appointments.telegram_id == tg_id).execute()