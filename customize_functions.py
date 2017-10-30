import pandas as pd

def load_dog_data():
    # Currently this website is now giving 404 error, fortunetly we saved the
    # the data inot a CSV file
    from os.path import isfile
    if isfile("dog_data.csv"):
        return pd.read_csv("dog_data.csv")

    import urllib2
    from bs4 import BeautifulSoup

    url = "http://www.sorteaze.com/dog-breed-personality--social-traits.html"
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page, "lxml")

    data = []
    for tr in soup.find_all('tr')[4:218]:
        tds = tr.find_all('td')
        data.append([td.text.encode('utf-8').strip() for td in tds])

    columns = [th.text.encode('utf-8').strip() for th in soup.find_all('th')]
    dog_personality = pd.DataFrame(data, columns=columns)

    url = "http://www.sorteaze.com/dog-breeds.html"
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page, "lxml")

    data = []
    for tr in soup.find_all('tr')[3:217]:
        tds = tr.find_all('td')
        data.append([td.text.encode('utf-8').strip() for td in tds])

    columns = [th.text.encode('utf-8').strip() for th in soup.find_all('th')]
    dog_traits = pd.DataFrame(data, columns=columns)
    dog_data = pd.merge(dog_personality, dog_traits, on='Breed Name', how='outer')
    dog_data.to_csv("dog_data.csv", index=False)
    return dog_data

def load_clean_dog_data():
    from os.path import isfile
    if isfile("dog_data_clean.csv"):
        return pd.read_csv("dog_data_clean.csv")
    dog_data = load_dog_data()
    dog_data.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    dog_data.loc[:, "BreedName"] = dog_data.BreedName.str.replace("- ", "")
    dog_data.drop(dog_data.columns[range(1, 4)+[9]+range(12, 15)], inplace=True, axis=1)
    # make new feature "Friendliness" base on the average
    dog_data["Friendliness"] = dog_data.iloc[:, 1:6].as_matrix().astype(int).mean(1)
    dog_data.loc[:, "Intelligent"] = dog_data.iloc[:, 8:10].as_matrix().astype(int).mean(1)
    dog_data.drop(["EasytoTrain"], axis=1, inplace=True)

    dog_data_clean = dog_data.drop(dog_data.columns[range(1, 6)], inplace=False, axis=1)
    dog_data_clean.to_csv("dog_data_clean_1.csv", index=False)
    return dog_data_clean
