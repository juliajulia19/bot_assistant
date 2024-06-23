import re
import pandas as pd

Name = ['иванова надежда васильевна', 'петров эдуард владимирович', 'коновалова светлана викторовна', 'попов максим степанович', 'катющева елена леонидовна', 'мордвин игорь иванович', 'черешнева екатерина петровна', 'кислов максим петрович', 'елисеев дмитрий алексеевич', 'любимова екатерина васильевна', 'васильев егор иванович', 'куница елена алексеевна', 'мартынов алексей семенович', 'дьяченко павел васильевич', 'васильева алена дмитриевна', 'дмитриев олег сергеевич', 'сергеева ольга васильевна', 'назарова марина витальевна', 'иванов максим владимирович', 'головин петр вячеславович']
Title = ['заместитель директора', 'бухгалтер', 'главный бухгалтер', 'системный администратор', 'специалист отдела кадров', 'старший специалист отдела кадров', 'дизайнер', 'главный дизайнер', 'специалист по связям с общественностью', 'smm-специалист', 'специалист отдела закупок', 'старший специалист отдела закупок', 'начальник отдела продаж', 'специалист отдела продаж', 'старший специалист отдела продаж', 'специалист отдела тендеров', 'специалист по работе с клиентами', 'старщий специалист по работе с клиентами', 'помощник руководителя', 'водитель']
Birthday = ['11.01.1990', '13.05.2000', '07.09.1987', '18.11.1991', '20.03.1995', '12.12.1997', '04.07.1987', '14.02.1990', '22.08.1993', '24.12.1985', '16.11.1997', '16.04.1993', '18.06.1988', '12.11.1983', '15.05.1988', '16.10.1991', '28.07.1996', '22.01.1991', '07.11.1989', '01.01.1979']
Phone = ['8921-567-44-03', '8916-555-16-87', '8911-112-65-74', '8911-312-76-87', '8921-657-43-43', '8950-156-01-12', '8952-544-87-57', '8905-663-11-23', '8906-587-08-01', '8904-001-45-54', '8911-034-12-34', '8911-346-54-01', '8921-407-29-03', '8950-034-12-84', '8927-809-24-75', '8951-024-38-87', '8921-911-46-87', '8903-759-95-24', '8911-345-87-45', '8917-847-04-46']
staff_df = pd.DataFrame({'name': Name, 'title': Title, 'birthday': Birthday, 'phone': Phone})

staff_df.to_csv('staff_df.csv')
print(staff_df.columns.tolist ())

filtered_staff = staff_df[staff_df['name'].str.contains('Иванова')]
if len(filtered_staff) > 0:
    print(filtered_staff)
else:
    print('не найдено')