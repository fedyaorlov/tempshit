# Этот скрипт сделан для получения данных из внешнего скрипта и на их основании считать цену автомобия.
import pandas as pd
import tensorflow as tf
import sqlite3 as sql

from joblib import load
from keras.models import load_model

NN_SCALER = load('models/scaler.pkl')
GOOD_LIST = ['хорош', 'отличн', 'качествен', 'дуже', 'очень', 'оригинал','заводск','нормальн','свеж','свіж',
             'отменн','відмін','ідеаль','идеал','гарном','гарний','гарна','максимал','предмаксима', 'ухожен',
             'бережн',' родн', ' рідн', 'доволен', 'довольны', 'прекрасн', 'чудес', 'шикарн', 'новый', 'новая', 'новое',
             'новые', 'нові', 'нова', 'новий', 'нове', 'прикрас', 'чудов', 'чист', 'впечатл', 'гарант', 'подтвержд', 'шикарн',
             'нового', 'добр', 'чисты','чист', 'якіс']
BAD_LIST = ['битая', 'бита', 'удар', 'вмятин', 'ржав', 'ушкодже', 'з дефект', 'с дефект', 'царап']
RF_MODEL = load('models/rf_model_final.pkl')
CAT_REGRESSOR = load('models/cat_regressor_final.pkl')
graph = tf.get_default_graph()
NN_MODEL = load_model('models/nn_reg_final.h5')
MY_DB = 'ave_prices.db'
conn = sql.connect(MY_DB, check_same_thread=False)
cursor = conn.cursor()


######################  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ  ##########################
def total_feature(dataframe, feats):
    #print('Начали ' + feats)
    indexes = dataframe.index
    features_list = []
    for idx in indexes:
        if idx % 5000 == 0 or idx == 0:
            # print(idx)
            pass
        feature = dataframe.loc[idx][feats]

        features = [feature]
        features_num = 0
        for feat in features:
            if type(feat) == float:
                continue
            length = len(feat.split(', '))
            features_num += length

        features_list.append(features_num)

    return features_list


def total_features(dataframe):
    indexes = dataframe.index
    features_list = []
    for idx in indexes:
        if idx % 200 == 0 or idx == 0:
            #print(idx)
            pass

        condition = dataframe.loc[idx]['Condition']
        safety = dataframe.loc[idx]['Safety']
        comfort = dataframe.loc[idx]['Comfort']
        multimedia = dataframe.loc[idx]['Multimedia']
        other = dataframe.loc[idx]['Other']

        features = [condition, safety, comfort, multimedia, other]
        features_num = 0
        for feat in features:
            if type(feat) == float:
                continue
            length = len(feat.split(', '))
            features_num += length

        features_list.append(features_num)

    return features_list

def first_reg(tags):
    if type(tags) is not float and 'Первая регистрация' in tags:
        first_reg = 1
    else:
        first_reg = 0
    return first_reg


def prignana(dataframe):
    indexes = dataframe.index
    ave_prices = []
    for idx in indexes:
        if idx % 1000 == 0 or idx == 0:
            #print(idx)
            pass

        try:
            desc = dataframe.loc[idx]['Description'].lower()
        except:
            desc = ''

        try:
            tags_list = dataframe.loc[idx]['Tags list'].lower()
        except:
            tags_list = ''

        if 'пригнан' in desc or 'пригнан' in tags_list:
            key = 1
        else:
            key = 0

        ave_prices.append(key)

    return ave_prices


def dtp(tags_list):
    dtp = 0
    if type(tags_list) != float:
        if 'ДТП' in tags_list or 'на ходу' in tags_list:
            dtp = 1

    return dtp


def personal_mark(dataframe):
    indexes = dataframe.index
    ave_prices = []
    counter = 0
    for idx in indexes:
        if idx % 1000 == 0 or idx == 0:
            #print(idx)
            pass

        try:
            desc = dataframe.loc[idx]['Description'].lower()
        except:
            desc = ''

        try:
            cond = dataframe.loc[idx]['Condition'].lower()
        except:
            cond = ''
        key = 2
        contin = True
        if 'не бит' in cond or 'не крашен' in cond:
            key = 3
            ave_prices.append(key)
        if key == 2:
            for condic in GOOD_LIST:
                if condic in desc:
                    key = 3
                    ave_prices.append(key)
                    contin = False
                    break
            if contin:
                for bad in BAD_LIST:
                    if bad in desc:
                        key = 1
                        ave_prices.append(key)
                        break


        if key == 2:
            ave_prices.append(key)
        counter += 1

    return ave_prices


def add_interactions(dataframe, features_list):
    for idx in range(len(features_list)):
        if idx < (len(features_list)-1):
            inter_list = features_list[idx+1:]
            for idx_f in range(len(inter_list)):
                first_feat = features_list[idx]
                second_feat = inter_list[idx_f]
                new_line =  first_feat + '_' + second_feat
                try:
                    dataframe[new_line] = dataframe[first_feat] * dataframe[second_feat]
                except:
                    dataframe[new_line] = float(dataframe[first_feat]) * float(dataframe[second_feat])
    return dataframe


def add_poly_df(dataframe, features_list, max_power):
    assert (type(max_power) == int), "Sorry, max_power variable is not integer. Try again"
    assert (max_power > 1), "Sorry, max_power should be bigger than 1"
    finish = max_power + 1

    for feature in features_list:
        for power in range(2, finish):
            new_line = feature + '^' + str(power)
            try:
                dataframe[new_line] = dataframe[feature] ** 2
            except:
                dataframe[new_line] = float(dataframe[feature]) ** 2

    return dataframe

def mech_transmission (transmission):
    if transmission == 'Ручная / Механика':
        return 1
    else:
        return 0
######################  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (КОНЕЦ)  ##########################


def prepare_data(data):
    for key in data:
        data[key] = [data[key]]
    table = pd.DataFrame.from_dict(data)

    table['Rating'] = 'N/A'
    table['Condition_n'] = total_feature(table, 'Condition')
    table['Safety_n'] = total_feature(table, 'Safety')
    table['Comfort_n'] = total_feature(table, 'Comfort')
    table['Multimedia_n'] = total_feature(table, 'Multimedia')
    table['Other_n'] = total_feature(table, 'Other')
    table['total_features'] = total_features(table)
    table['first_reg'] = table['Tags list'].apply(first_reg)
    table['model_year'] = list(table['Model'])[0] + ', ' + list(table['Year'])[0]
    table['prignana'] = prignana(table)
    table['dpt'] = table['Tags list'].apply(dtp)
    table['my_estimation'] = personal_mark(table)
    table['Price'] = 6000
    table = table[
        ['Price', 'Producer', 'Model', 'Year', 'Mileage', 'Volume', 'Fuel', 'Transmission', 'Powertrain', 'Description',
         'Body',
         'Region', 'Rating', 'Condition', 'Safety', 'Comfort', 'Multimedia', 'Other', 'Tags list', 'Doors', 'Seats',
         'total_features',
         'model_year', 'first_reg', 'prignana', 'dpt', 'Condition_n', 'Safety_n', 'Comfort_n', 'Multimedia_n',
         'Other_n', 'my_estimation']]
    # Приведение названия фич к их правильному дипломному виду
    all_lower = ['Price', 'Year', 'Mileage', 'Volume', 'Doors', 'Seats', 'total_features', 'model_year', 'Condition_n',
                 'Safety_n', 'Comfort_n',
                 'Multimedia_n', 'Other_n', 'Constant', ]
    all_lower = [word.lower() for word in all_lower]

    old_keys = table.keys()
    new_keys = []
    for key in old_keys:
        if key.lower() in all_lower:
            new_keys.append(key.lower())
        else:
            new_keys.append(key.upper())
    table.columns = new_keys

    try:
        cursor.execute('SELECT MY_ave from MY_table where model_year="{}"'.format(list(table['model_year'])[0]))
        rows = cursor.fetchall()
        model_year = float(rows[0][0])
    except:
        cursor.execute('SELECT price from model_table where model="{}"'.format(list(table['MODEL'])[0]))
        rows = cursor.fetchall()
        model_year = float(rows[0][0])

    cursor.execute('SELECT Prod_ave from prod_table where Producer="{}"'.format(list(table['PRODUCER'])[0]))
    rows = cursor.fetchall()
    producer = float(rows[0][0])

    cursor.execute('SELECT Reg_ave from reg_table where Region="{}"'.format(list(table['REGION'])[0]))
    rows = cursor.fetchall()
    region = float(rows[0][0])

    # Для рандомфореста обрабатываем
    # for RF
    rf_x = table.copy()
    encoded_columns = ['POWERTRAIN_Передний', 'POWERTRAIN_Полный', 'BODY_Кабриолет', 'BODY_Купе',
                       'BODY_Легковой фургон (до 1,5 т)',
                       'BODY_Лимузин', 'BODY_Лифтбек', 'BODY_Минивэн', 'BODY_Пикап', 'BODY_Родстер', 'BODY_Седан',
                       'BODY_Универсал',
                       'BODY_Хэтчбек', 'FUEL_dizel', 'FUEL_drugoe', 'FUEL_elektro', 'FUEL_gaz', 'FUEL_gaz-benzin',
                       'FUEL_gaz-metan', 'FUEL_gaz-propan-butan', 'FUEL_gibrid', 'TRANSMISSION_Адаптивная',
                       'TRANSMISSION_Вариатор', 'TRANSMISSION_Ручная / Механика', 'TRANSMISSION_Типтроник']

    for col in encoded_columns:
        rf_x[col] = [0]

    powertrains = [item[11:] for item in list(rf_x) if 'POWERTRAIN' in item]
    bodies = [item[5:] for item in list(rf_x) if 'BODY' in item]
    fuels = [item[5:] for item in list(rf_x) if 'FUEL' in item]
    trans = [item[13:] for item in list(rf_x) if 'TRANSMISSION' in item]

    if rf_x['POWERTRAIN'][0] in powertrains:
        rf_x['POWERTRAIN_' + rf_x['POWERTRAIN']] = [1]
    if rf_x['BODY'][0] in bodies:
        rf_x['BODY_' + rf_x['BODY']] = [1]
    if rf_x['FUEL'][0] in fuels:
        rf_x['FUEL_' + rf_x['FUEL']] = [1]
    if rf_x['TRANSMISSION'][0] in trans:
        rf_x['TRANSMISSION_' + rf_x['TRANSMISSION']] = [1]

    cols_to_del = ['MODEL', 'DESCRIPTION', 'RATING', 'CONDITION', 'SAFETY', 'COMFORT', 'MULTIMEDIA', 'OTHER',
                   'TAGS LIST']
    rf_x = rf_x.drop(columns=cols_to_del)

    rf_x['PRODUCER'] = [producer]
    rf_x['model_year'] = [model_year]
    rf_x['REGION'] = [region]
    rf_x = rf_x.drop(columns=['price'])
    rf_x = rf_x[['PRODUCER', 'year', 'mileage', 'volume', 'REGION', 'doors', 'seats', 'total_features', 'model_year',
                 'FIRST_REG', 'PRIGNANA', 'DPT', 'condition_n', 'safety_n', 'comfort_n', 'multimedia_n', 'other_n',
                 'MY_ESTIMATION', 'POWERTRAIN_Передний', 'POWERTRAIN_Полный', 'BODY_Кабриолет', 'BODY_Купе',
                 'BODY_Легковой фургон (до 1,5 т)', 'BODY_Лимузин', 'BODY_Лифтбек', 'BODY_Минивэн', 'BODY_Пикап',
                 'BODY_Родстер', 'BODY_Седан', 'BODY_Универсал', 'BODY_Хэтчбек', 'FUEL_dizel', 'FUEL_drugoe',
                 'FUEL_elektro', 'FUEL_gaz', 'FUEL_gaz-benzin', 'FUEL_gaz-metan', 'FUEL_gaz-propan-butan',
                 'FUEL_gibrid', 'TRANSMISSION_Адаптивная', 'TRANSMISSION_Вариатор', 'TRANSMISSION_Ручная / Механика',
                 'TRANSMISSION_Типтроник']]

    # Для градиентного бустинга обрабатываем
    x_gb = table.copy()
    x_gb['model_year'] = [model_year]
    features_to_interact = ['year', 'mileage', 'volume', 'model_year']
    x_gb = add_interactions(x_gb, features_to_interact)
    # print('ДОбавляем полином в GB')
    features_to_poly = ['year', 'mileage', 'volume', 'total_features', 'model_year']
    max_power = 2
    x_gb = add_poly_df(x_gb, features_to_poly, max_power)
    x_gb = x_gb[
        ['PRODUCER', 'year', 'mileage', 'volume', 'FUEL', 'TRANSMISSION', 'POWERTRAIN', 'BODY', 'REGION', 'doors',
         'seats', 'total_features', 'model_year', 'FIRST_REG', 'PRIGNANA', 'DPT', 'condition_n', 'safety_n',
         'comfort_n', 'multimedia_n', 'other_n', 'MY_ESTIMATION', 'year_mileage', 'year_volume', 'year_model_year',
         'mileage_volume', 'mileage_model_year', 'volume_model_year', 'year^2', 'mileage^2', 'volume^2',
         'total_features^2', 'model_year^2']]

    # Для нейронки обрабатывае
    x_nn = table.copy()
    for col in encoded_columns:
        x_nn[col] = [0]

    powertrains = [item[11:] for item in list(x_nn) if 'POWERTRAIN' in item]
    bodies = [item[5:] for item in list(x_nn) if 'BODY' in item]
    fuels = [item[5:] for item in list(x_nn) if 'FUEL' in item]
    trans = [item[13:] for item in list(x_nn) if 'TRANSMISSION' in item]

    if x_nn['POWERTRAIN'][0] in powertrains:
        x_nn['POWERTRAIN_' + x_nn['POWERTRAIN']] = [1]
    if x_nn['BODY'][0] in bodies:
        x_nn['BODY_' + x_nn['BODY']] = [1]
    if x_nn['FUEL'][0] in fuels:
        x_nn['FUEL_' + x_nn['FUEL']] = [1]
    if x_nn['TRANSMISSION'][0] in trans:
        x_nn['TRANSMISSION_' + x_nn['TRANSMISSION']] = [1]
    x_nn['PRODUCER'] = [producer]
    x_nn['model_year'] = [model_year]
    x_nn['REGION'] = [region]

    features_to_interact = ['year', 'mileage', 'volume', 'model_year']
    features_to_poly = ['year', 'mileage', 'volume', 'model_year']
    max_power = 2
    x_nn = add_interactions(x_nn, features_to_interact)
    # print('ДОбавляем полином в NN')
    x_nn = add_poly_df(x_nn, features_to_poly, max_power)
    x_nn['PRODUCER'] = [producer]
    x_nn['model_year'] = [model_year]
    x_nn['REGION'] = [region]
    x_nn['TRANSMISSION'] = x_nn['TRANSMISSION'].apply(mech_transmission)
    x_nn = x_nn[
        ['PRODUCER', 'year', 'mileage', 'volume', 'TRANSMISSION', 'REGION', 'seats', 'total_features', 'model_year',
         'PRIGNANA', 'DPT', 'POWERTRAIN_Передний', 'POWERTRAIN_Полный', 'BODY_Кабриолет', 'BODY_Купе',
         'BODY_Легковой фургон (до 1,5 т)', 'BODY_Лимузин', 'BODY_Лифтбек', 'BODY_Минивэн', 'BODY_Пикап',
         'BODY_Родстер', 'BODY_Седан', 'BODY_Универсал', 'BODY_Хэтчбек', 'year_mileage', 'year_volume',
         'year_model_year', 'mileage_volume', 'mileage_model_year', 'volume_model_year', 'year^2', 'mileage^2',
         'volume^2', 'model_year^2']]

    x_nn = NN_SCALER.transform(x_nn)

    return rf_x, x_gb, x_nn


def predict_price(data):
    x_rf, x_gb, x_nn = prepare_data(data)
    rf_pred = RF_MODEL.predict(x_rf)[0]
    cat_pred = CAT_REGRESSOR.predict(x_gb)[0]
    with graph.as_default():
        nn_pred = NN_MODEL.predict(x_nn)[0][0]

    print('RF: ', str(rf_pred))
    print('CATBOSST: ', str(cat_pred))
    print('NN: ', str(nn_pred))

    price = (rf_pred + cat_pred + nn_pred)/3
    return [rf_pred, cat_pred, nn_pred, price]
