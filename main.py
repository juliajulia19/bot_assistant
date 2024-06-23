import logging
import re
import random
import pandas as pd

from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.files import JSONStorage
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import ReplyKeyboardMarkup

from newsMK import news_df
from assist_database import Procurement

# Configure logging

logging.basicConfig(level=logging.DEBUG)

API_TOKEN = "7125015588:AAFoN50jdrxbxZRW3bjG_BM7EY1l-DnYjJM"
storage = JSONStorage("storage.json")
bot = Bot(token=API_TOKEN)

dp = Dispatcher(bot, storage=storage)

class States(StatesGroup):
    main_state = State()
    news_state = State()
    staff_state = State()
    procurement_state_1 = State() #для ввода наименования товара
    procurement_state_2 = State() #для ввода количества
    chedule_state = State()

staff_df = pd.read_csv('staff_df.csv')

#основное меню по команде /start

@dp.message_handler(commands=['start', 'help'], state="*")
async def send_welcome(message: types.Message, state: FSMContext):
    await States.main_state.set()
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("почитать новости", "уточнить информацию о сотрудниках", "заказать канцелярию", "внести встречу в расписание")
    with open('assistant.png', 'rb') as photo:
        await message.answer_photo(photo, caption="Приветствую! Меня зовут Мария. Я Ваш виртальный ассистент. Пожалуйста выберите нужный раздел, нажав кнопку ниже",
                        reply_markup=markup)
    logging.info(f"Message from {message.from_user.username}: {message.text}")

#возвращение в главное меню

@dp.message_handler(regexp='.*главное\sменю.*', state="*")
async def main_menu(message: types.Message, state: FSMContext):
    await States.main_state.set()
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("почитать новости", "уточнить информацию о сотрудниках", "заказать канцелярию", "внести встречу в расписание")
    await message.answer("Пожалуйста выберите нужный раздел, нажав кнопку ниже",
                        reply_markup=markup)
    logging.info(f"Message from {message.from_user.username}: {message.text}")

#раздел новости

@dp.message_handler(regexp='.*(почитать\sновости)|(следующая\sновость).*', state=States.main_state)
async def send_news(message: types.Message, state: FSMContext):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("следующая новость", "главное меню")
    random_number = random.randint(0, 153)
    await message.answer(f'{news_df.iloc[random_number, 1]} \n'  #выдаем случайную новость из датафрейма
                         f'Ссылка на источник: {news_df.iloc[random_number, 0]} \n'
                         f' \n'
                         f'Нажмите кнопку "Следующая новость", чтобы ознакомиться с другими новостями или вернитесь в главное меню',
                         reply_markup=markup)
    logging.info(f"Message from {message.from_user.username}: {message.text}")

#раздел информация о сотрудниках

@dp.message_handler(regexp='.*уточнить\sинформацию\sо\sсотрудниках.*', state=States.main_state)
async def send_news(message: types.Message, state: FSMContext):
    await States.staff_state.set()
    await message.answer("Напишите фамилию или должность сотрудника, который Вас интересует", #будет искать только с большой буквы, подумай как реализовать поиск и с маленькой
                         reply_markup=types.ReplyKeyboardRemove())
    logging.info(f"Message from {message.from_user.username}: {message.text}")

@dp.message_handler(state=States.staff_state)
async def find_staff_info(message: types.Message, state: FSMContext):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("главное меню")
    filtered_staff_name = staff_df[staff_df['name'].str.contains(message.text.lower())]
    filtered_staff_title = staff_df[staff_df['title'].str.contains(message.text.lower())]
    if len(filtered_staff_name) > 0 and len(filtered_staff_title) == 0:
        await message.answer(f'{filtered_staff_name.iloc[0, 1]} \n'
                             f'{filtered_staff_name.iloc[0, 2]} \n'
                             f'дата рождения: {filtered_staff_name.iloc[0, 3]} \n'
                             f'номер телефона: {filtered_staff_name.iloc[0, 4]} \n',
                             reply_markup=markup)
    elif len(filtered_staff_name) == 0 and len(filtered_staff_title) > 0:
        await message.answer(f'{filtered_staff_title.iloc[0, 1]} \n'
                             f'{filtered_staff_title.iloc[0, 2]} \n'
                             f'дата рождения: {filtered_staff_title.iloc[0, 3]} \n'
                             f'номер телефона: {filtered_staff_title.iloc[0, 4]}',
                             reply_markup=markup)
    else:
        await message.answer("Такого сотрудника нет в базе, пожалуйста попробуйте изменить запрос или вернитесь в главное меню",
                             reply_markup=markup)

#раздел заказ канцелярии

@dp.message_handler(regexp='.*(заказать\sканцелярию)|(продолжить).*', state=States.main_state)
async def send_news(message: types.Message, state: FSMContext):
    await States.procurement_state_1.set()
    await message.answer("Пожалуйста введите наименование товара",
                         reply_markup=types.ReplyKeyboardRemove())
    logging.info(f"Message from {message.from_user.username}: {message.text}")

@dp.message_handler(regexp='^[A-zА-яЁё\s]+$', state=States.procurement_state_1) #регулярка на только буквы и пробел
async def send_news(message: types.Message, state: FSMContext):
    await States.procurement_state_2.set()
    await message.answer("Пожалуйста укажите необходимое количество, введя только цифру")
    async with state.proxy() as data:
        data['product'] = [message.text][0] #выбираем только одно первое сообщение, иначе в базу сохраняется список
    logging.info(f"Message from {message.from_user.username}: {message.text}")

@dp.message_handler(regexp='^\d+$', state=States.procurement_state_2) #регулярка на только цифры
async def send_news(message: types.Message, state: FSMContext):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("продолжить", "главное меню")
    await message.answer("Информация внесена в базу. Ответственный сотрудник закажет необходимый товар. Если необходимо заказать, что-то еще нажмите кнопку 'Продолжить' или вернитесь в главное меню",
                         reply_markup=markup)
    await States.main_state.set()
    async with state.proxy() as data:
        data['quantity'] = [message.text][0]
    Procurement.create(customer=message.from_user.full_name, telegram_id=message.from_user.id, product=data['product'],
                        quantity=data['quantity'])
    logging.info(f"Message from {message.from_user.username}: {message.text}")

@dp.message_handler(state=States.procurement_state_2) #если введено не число
async def send_news(message: types.Message, state: FSMContext):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("главное меню")
    await message.answer("Неверный запрос пожалуйста введите именно цифру",
                         reply_markup=markup)
    logging.info(f"Message from {message.from_user.username}: {message.text}")

#отлов нераспознанных запросов
@dp.message_handler(state="*")
async def unknown(message: types.Message, state: FSMContext):
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("главное меню")
    await message.answer('Нераспознанный запрос или вы ошиблись при вводе данных \n'
                         'Чтобы вернуться в главное меню нажмите кнопку ниже',
                         reply_markup=markup)
    logging.info(f"Message from {message.from_user.username}: {message.text}")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)