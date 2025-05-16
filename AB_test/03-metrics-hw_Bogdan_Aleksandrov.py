"""Домашнее задание.

Написать функцию вычисляющую метрику пользователей лифта.

ОПИСАНИЕ СИТУАЦИИ
Допустим у нас есть 10-ти этажное здание, в котором есть один лифт вместимостью 10 человек.
На каждом этаже есть кнопка вызова лифта. Когда человеку нужно попасть с этажа Х на этаж У, он нажимает кнопку вызова
лифта, ждёт когда лифт приедет, заходит в лифт, нажимает на кнопку этажа У, и когда лифт приезжает на нужный
этаж - выходит. В час-пик лифт может долго не приезжать, если это происходит, то человек может пойти пешком по лестнице
на нужный ему этаж.

АБ ЭКСПЕРИМЕНТ
Мы хотим изменить алгоритм работы лифта и с помощью АБ теста оценить как это повлияет на время, затрачиваемое людьми
на перемещение между этажами. В качестве метрики будем использовать некоторую статистику от эмпирической функции
распределения затрачиваемого времени пользователей.

МЕТРИКА ЭКСПЕРИМЕНТА
Затрачиваемое время T определим так:
- если человек дождался лифта, то T = t2 - t1, где t2 - время прибытия на целевой этаж, а t1 - время вызова лифта.
- если человек не дождался лифта и пошёл пешком, то T = 2 * (t2 - t1), где t2 - время когда решил пойти пешком, t1 - 
время вызова лифта.

ОТКУДА ДАННЫЕ
Данные генерируются с помощью эмуляции работы лифта и случайного генерирования пользователей.
Для простоты время разбито на интервалы по 10 секунд.

В каждый интервал времени лифт может совершить одно из 3 действий: спуститься на 1 этаж вниз, подняться на 1 этаж 
вверх, произвести посадку/высадку на текущем этаже.

В каждый интервал времени с некоторой вероятностью генерируются люди, которые обладают следующими свойствами:
- текущий этаж, с которого хочет уехать;
- целевой этаж, куда хочет приехать;
- начальное время, когда начал ждать лифт;
- максимальное время ожидания, если не зайдёт в лифт до этого момента, то пойдёт пешком.

ОПИСАНИЕ ДАННЫХ
У нас есть табличка с логами лифта, в которой есть следующие атрибуты:
- time - время в секундах.
- action - состояние лифта в следующие 10 секунд. OPEN - стоит открытый, UP - едет вверх, DOWN - едет вниз.
- user_out - количество вышедших человек
- user_in - количество вошедших человек
- user_total - количество человек в лифте
- floor - текущий этаж
- calls - список вызовов. Вызов описывается парой значений - время вызова и этаж, на который был вызван лифт.
- orders - список заказов, на какие этажи нажимали пользователи, зашедшие в лифт. Аналогично содержит список пар - 
время заказа и целевой этаж.

ЗАДАНИЕ
Нужно написать функцию, которая принимает на вход таблицу pd.DataFrame с логами лифта и возвращает множество значений
метрик пользователей. Метрика описана выше в разделе МЕТРИКА ЭКСПЕРИМЕНТА.

ПРИМЕР
Рассмотрим кусок данных. Тут пользователь вызвал лифт при t1=10, и доехал на нужный этаж при t2=40, значение
метрики для него будет равно t2 - t1 = 30.

time  | action | user_out | user_in | user_total | floor |    calls    |    orders
--------------------------------------------------------------------------------------
0     | open   | 0        | 0       | 0          | 1     |    []       |    []    
10    | up     | 0        | 0       | 0          | 1     | [(10, 2)]   |    []    
20    | open   | 0        | 1       | 1          | 2     |    []       | [(20, 1)]    
30    | down   | 0        | 0       | 1          | 2     |    []       | [(20, 1)]    
40    | open   | 1        | 0       | 0          | 1     |    []       |    []      

ОЦЕНИВАНИЕ
По данным из вашей функции и по истинным значениям метрики будут построены эмпирические функция распределения.
Далее будет вычислено максимальное отличие между полученными ЭФР (аналогично статистике критерия Колмогорова).
Чем меньше отличие D, тем выше балл.
- D <= 0.1 - 10 баллов
- D <= 0.13 - 9 баллов
- D <= 0.16 - 8 баллов
- D <= 0.19 - 7 баллов
и так далее с шагом 0.03.
"""

import pandas as pd


data = pd.read_csv('df_elevator_logs.csv')


def calculate_metrics(data: pd.DataFrame):
    """Вычисляет значения метрики пользователей.
    
    data - таблица с логами лифта

    return - список значений метрики
    """
    stats = []
    prev_row = pd.Series({'time':0, 'action':'open', 'user_out':0, 'user_in':0, 'user_total':0, 'floor':1, 'calls':'[]', 'orders':'[]'})
    floor_to_when_called = dict([(i, []) for i in range(1, 11)])
    floors_unknown = dict([(i, set()) for i in range(1, 11)])
    unknown_floor_to_data = dict([(i, []) for i in range(1, 11)])

    for _, row in data.iterrows():
        if row.action == 'open':
            used_call = list(set(eval(prev_row.calls)) - set(eval(row.calls)))
            called_time = used_call[0][0] if len(used_call) == 1 else row.time

            if row.user_in > 0:
                added_orders = set(eval(row.orders)) - set(eval(prev_row.orders))
                if len(eval(row.orders)) == 1:
                    floor_to_when_called[eval(row.orders)[0][1]].append((called_time, row.user_in))
                else:
                    user_in_left = row.user_in
                    for added_order in added_orders:
                        floor_to_when_called[added_order[1]].append((added_order[0], 1))
                        user_in_left -= 1
                    
                    if user_in_left:
                        potential_floors_to_go = set()
                        for tmp_o in eval(row.orders):
                            potential_floors_to_go.add(tmp_o[1])
                        for order in eval(row.orders):
                            floors_unknown[order[1]].add(row.floor)
                        unknown_floor_to_data[row.floor].append([row.time, user_in_left, potential_floors_to_go])
            elif len(used_call) == 1:
                stats.append(2 * 90) # mean weight time
            
            if row.user_out > 0:
                user_out_left = row.user_out
                for call in floor_to_when_called[row.floor]:
                    stats.extend([row.time - call[0]] * call[1])
                    user_out_left -= call[1]
                
                if user_out_left:
                    potential_users_floors = list(floors_unknown[row.floor])
                    floor_diffs = []
                    for tmp_f in potential_users_floors:
                        floor_diffs.append(abs(tmp_f - row.floor))
                    users_sorted_by_floor = sorted(range(len(potential_users_floors)), key=lambda x: floor_diffs[x],reverse=True)
                    for tmp_f in users_sorted_by_floor:
                        for i, data_unknown in enumerate(unknown_floor_to_data[potential_users_floors[tmp_f]]):
                            if data_unknown[1] <= user_out_left:
                                stats.extend([row.time - data_unknown[0]] * data_unknown[1])
                                user_out_left -= data_unknown[1]
                                unknown_floor_to_data[potential_users_floors[tmp_f]][i][1] = 0
                            else:
                                stats.extend([row.time - data_unknown[0]] * user_out_left)
                                unknown_floor_to_data[potential_users_floors[tmp_f]][i][1] -= user_out_left
                                user_out_left = 0
                                break

                        if user_out_left:
                            unknown_floor_to_data[potential_users_floors[tmp_f]] = []
                        else:
                            if unknown_floor_to_data[potential_users_floors[tmp_f]][i][1] == 0:
                                unknown_floor_to_data[potential_users_floors[tmp_f]] = unknown_floor_to_data[potential_users_floors[tmp_f]][i+1:]
                            else:
                                unknown_floor_to_data[potential_users_floors[tmp_f]] = unknown_floor_to_data[potential_users_floors[tmp_f]][i:]
                            break
            
            cur_orders_floors = set(map(lambda x: x[1], eval(row.orders)))
            for tmp_f in unknown_floor_to_data:
                left_indeces = []
                for i, v in enumerate(unknown_floor_to_data[tmp_f]):
                    common_ords = list(cur_orders_floors.intersection(v[2]))
                    if len(common_ords) == 1:
                        floor_to_when_called[common_ords[0]].append((v[0], v[1]))
                    else:
                        unknown_floor_to_data[tmp_f][i][2].discard(row.floor)
                        if len(unknown_floor_to_data[tmp_f][i][2]) == 1:
                            for now_known_f in unknown_floor_to_data[tmp_f][i][2]:
                                floor_to_when_called[now_known_f].append((v[0], v[1]))
                        else:
                            left_indeces.append(i)
                new_arr = []
                for i, v in enumerate(unknown_floor_to_data[tmp_f]):
                    if i in left_indeces:
                        new_arr.append(v)

                unknown_floor_to_data[tmp_f] = new_arr
            
            for tmp_f in floors_unknown:
                empty_floors_from = []
                for fl in floors_unknown[tmp_f]:
                    if len(unknown_floor_to_data[fl]) == 0:
                        empty_floors_from.append(fl)
                for fl in empty_floors_from:
                    floors_unknown[tmp_f].discard(fl)

            floor_to_when_called[row.floor] = []
            floors_unknown[row.floor] = set()
        
        prev_row = row

    return stats
