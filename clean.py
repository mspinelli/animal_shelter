import pandas as pd
import numpy as np
import MyHolidays
from scipy.stats import bernoulli
from customize_functions import load_clean_dog_data
import re
import itertools
from functools import wraps
import sys
import webcolors


tx_holidays = MyHolidays.MyHolidays(state="TX", years=[2015])
units = {"year": 365, "month": 31, "day": 1, "week": 7}
bins = [-5, 0, 5 * units["month"], 12 * units["month"], 3*units["year"],
        6 * units["year"], 10*units["year"], 50 * units["year"]]
labels = ["Unknown", "Infant", "Puppy/Kitten", "Young Adult", "Adult", "Senior", "Geriatric"]
im_p = .53
dog_data_clean = load_clean_dog_data()

color_pattern = dict(
    Tabby='Tabby',
    Agouti='Tabby',
    Tiger='Tabby',
    Point='Point',
    Tortie='TriColor',
    Calico='TriColor',
    Torbie='TriColor',
    Brindle='TriColor',
    Smoke='Smoke',
    Merle='Merle')

color_synonym =dict(
    Flame='Orange',
    Apricot='Orange',
    Red='Red',
    Ruddy='Red',
    Gold='Yellow',
    Cream='Tan',
    Buff='Tan',
    Fawn='Tan',
    Liver='Brown',
    Seal='Brown',
    Sable='Brown',
    Chocolate='Brown',
    Blue='Grey',
    Lilac='Grey',
    goldenrod="yellow",
    darkorange="orange",
    pink="salmon",
    peru="brown",
    saddlebrown="brown",
    sandybrown="brown",
    yellowgreen = "yellow",
    olive="darkolivegreen",
    chocolate="brown",
    darkgray="grey",
    gainsboro="grey",
    mistyrose="salmon"
)
color = {"goldenrod": "yellow",
         "darkorange": "orange",
         "pink" : "salmon",
         "peru" : "brown",
         "saddlebrown" : "brown",
         "sandybrown" : "brown",
         "yellowgreen"
         "olive" : "darkolivegreen",
         "chocolate" : "brown",
         "darkgray" : "grey",
         "gainsboro" : "grey"}

main_color = [
    'Black',
    'Brown',
    'Yellow',
    'Gray',
    'Orange',
    'Pink',
    'Red',
    'Silver',
    'Tan',
    'White']

def memoize(func):
    cache = {}

    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrap


@memoize
def try_permutations(tokens):
    min_count = sys.maxint
    best_string = None
    for i in range(len(tokens)):
        candidate = itertools.permutations(tokens, i + 1)
        for c in candidate:
            string = ' '.join(x for x in c)
            count = len(dog_data_clean[dog_data_clean.BreedName.str.contains(
                string, case=False)])
            if count == 1:
                return string
            if count < min_count and count != 0:
                min_count, best_string = count, string
    return best_string


def add_dog_data(x):
    name = x.Breed.replace(" Mix", "")
    breeds = name.split("/")

    matches = pd.DataFrame()
    for br in breeds:
        best_string = try_permutations(frozenset(br.split(" ")))
        if best_string:
            matches = pd.concat(
                [matches,
                 dog_data_clean[dog_data_clean.BreedName.str.contains(
                     best_string, case=False)]
                 ])

    return pd.Series(matches.iloc[:, 1:]
                     .as_matrix()
                     .astype(int)
                     .mean(axis=0)
                     .round(0))

def is_holiday(dt):
    if (dt.month in [1, 12, 7, 8]) or (dt.date() in tx_holidays) or dt.weekday() in [5, 6]:
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

def mix_type(x):
    if 'mix' in x.lower():
        return 'Mix'
    elif '/' in x:
        return 'Cross'
    else:
        return 'Pure'

def hair(x):
    if 'Short' in str(x):
        return 'Short'
    elif 'Medium' in str(x):
        return 'Medium'
    elif 'Long' in str(x):
        return 'Long'
    else:
        return np.nan


def reduce_colors(color):
    result = []
    for c in [c.strip() for c in re.split('/| ', color)]:
        if c in main_color:
            result.append(c)
        elif c in color_synonym.keys():
            result.append(color_synonym[c])
    return 'Unknown' if not result else '/'.join(np.sort(list(set(result))))

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name

@memoize
def get_blend_color(colorstring):
    colors = colorstring.split("/")
    pallette = np.array([np.hstack(webcolors.name_to_rgb(c)) for c in colors])
    blend = np.mean(pallette, axis=0)
    return get_colour_name(blend)

def color_name(colorstring):
    if (colorstring == "Unknown"):
        return np.nan
    cs = get_blend_color(colorstring)
    if cs in color_synonym.keys():
        return color_synonym[cs]
    return cs

def reduce_pattern(color):
    # result = None
    for c in [c.strip() for c in re.split('/| ', color)]:
        if c in color_pattern.keys():
            return color_pattern[c]
    return 'BiColor' if '/' in color else 'Solid'


def multinomial_impute(series, data):
    """
    :param series: a pandas.Series that is filtered to be very similar to missing values properties. 
    We count per class probability in this series.
    :param data: the original dataframe we want to impute changes on.
    return: a numpy array which has the same size as the number of null values in the series.
    """
    vc = series.value_counts(normalize=True)
    values = vc.axes[0].tolist()
    ps = vc.tolist()

    assert len(values) == len(ps)

    sample = np.random.multinomial(1, ps, size=data[series.name].isnull().sum())
    vs = np.array([values[i] for i in sample.argmax(axis=1)])
    # animals.loc[animals[series.name].isnull(), series.name] = vs
    data.loc[data[series.name].isnull(), series.name] = vs

def make_features(status="test"):
    if(status=="test"):
        animals = pd.read_csv("test.csv.gz", compression='gzip').iloc[:,1:]
    else: # train data
        animals = pd.read_csv("train.csv.gz", compression='gzip')
        print "here!"
        animals.loc[3174, "SexuponOutcome"] = "Neutered Male"
        numofcols = animals.shape[-1]

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
    animals.drop("Timestamps", axis=1, inplace=True)

    animals["AgeinDays"] = animals.AgeuponOutcome.map(get_age_in_days, na_action="ignore")
    animals.AgeinDays.fillna(value=-1, inplace=True)
    animals["LifeStage"] = pd.Categorical(pd.cut(animals.AgeinDays, bins=bins, labels=labels), categories=labels)

    animals["BernoulliSex"] = animals["SexuponOutcome"].map(unknown_sex, na_action="ignore")
    # Now adding dog data

    animals[dog_data_clean.columns[1:]] = (animals[animals.AnimalType == 'Dog'].apply(add_dog_data, axis=1))
    animals['MixType'] = animals["Breed"].map(mix_type)
    animals.loc[animals.AnimalType == "Cat", 'AvgWeight(pounds)'] = 9
    animals.loc[animals.AnimalType == "Cat", 'SizeScore(1to5)'] = 1
    animals.loc[animals.AnimalType == "Cat", "Intelligent"] = 3
    animals.loc[animals.AnimalType == "Cat", "Friendliness"] = 3

    animals['Hair'] = animals["Breed"].map(hair)
    animals['ReducedColor'] = animals.Color.map(reduce_colors)
    animals["BlendedColor"] = animals.ReducedColor.map(color_name, na_action="ignore")
    animals['ReducedPattern'] = animals.Color.map(reduce_pattern)
    animals.drop("ReducedColor", inplace=True, axis=1)

    # putting any existing different feature columns in the back if making train
    if (status=="train"):
        test = pd.read_csv("test.csv.gz", compression='gzip').iloc[:,1:]
        colmask = np.array([idx in test.columns for idx in animals.columns[:numofcols]])
        # print colmask
        cols = animals.columns[:numofcols][colmask].tolist() + \
               animals.columns[numofcols:].tolist() + \
               animals.columns[:numofcols][~colmask].tolist()
        animals = animals[cols]

    return animals

def impute_features(df):
    # gt = pd.read_csv("train.csv.gz", compression='gzip')
    columns_of_interest = range(7, 26)

    for col in df.columns[columns_of_interest]:
        # all BlendedColor missings are bc the records are not color
        if df[col].isnull().sum() == 0:
            continue
        if col == "BlendedColor":
            df.loc[df[col].isnull(), col] = "Not_clear"
        elif col == "Hair":
            multinomial_impute(df.Hair, data=df)
        else:
            multinomial_impute(df.loc[df.AnimalType=="Dog", col], data=df)
    return df

