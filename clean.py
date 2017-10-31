import pandas as pd
import numpy as np
import MyHolidays
from scipy.stats import bernoulli
from customize_functions import load_clean_dog_data

tx_holidays = MyHolidays.MyHolidays(state="TX", years=[2015])
units = {"year" : 365, "month":31, "day":1, "week":7}
bins = [-5, 0, 5 * units["month"], 12 * units["month"], 3*units["year"],
        6 * units["year"], 10*units["year"], 20 * units["year"]]
labels = ["Unknown", "Infant", "Puppy/Kitten", "Young Adult", "Adult", "Senior", "Geriatric"]


def is_holiday(dt):
    if (dt.month in [1, 12, 7, 8]) or (dt.date() in tx_holidays) or dt.weekday() in [5,6]:
        return True
    return False


def unknown_sex(sexstring):
    if sexstring == "Unknown":
        sex = bernoulli.rvs(im_p)
        if sex == 1:
            return "Intact Male"
        else:
            return "Intact Female"
    else:
        return sexstring

def get_age_in_days(agestring):
    tokens = agestring.split(" ")
    unit = tokens[1].replace("s","")
    days = int(tokens[0]) * units[unit]
    # return days
    return days if days != 0 else -1

def make_features():
    animals = pd.read_csv("test.csv", compression='gzip')

    animals['HasName'] = animals.Name.map(lambda x: 0 if x == 'X' else 1, na_action='ignore')
    animals['HasName'].fillna(value=0, inplace=True)

    animals['Timestamps'] = pd.to_datetime(animals['DateTime'])
    animals['Year'] = animals.Timestamps.dt.year
    animals['Month'] = animals.Timestamps.dt.month
    animals['Weekday'] = animals.Timestamps.dt.weekday
    animals['Hour'] = animals.Timestamps.dt.hour
    animals['WeekofYear'] = animals.Timestamps.dt.weekofyear
    animals['DayofMonth'] = animals.Timestamps.dt.day
    animals["isHoliday"] = animals.Timestamps.apply(is_holiday)

    animals["AgeinDays"] = animals.AgeuponOutcome.map(get_age_in_days, na_action="ignore")
    animals["LifeStage"] = pd.Categorical(pd.cut(animals.AgeinDays, bins=bins, labels=labels), categories=labels)

    # skipping bernoulli sex

    dog_data_clean = load_clean_dog_data()

if __name__ == "__main__":
    make_features()