import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def get_data(date_range, city_id, city_name):
    # if there is no local data, read data through the web and save them locally

    # loading local data
    try:
        local_data = pd.read_pickle(city_name)

        print('local data loaded')
        return local_data

    # loading online data
    except IOError:
        # initializing the series to store data
        new_data = pd.Series(dtype='float32')

        # getting every day's data in the range
        for day in date_range:
            day = day.strftime('%F')
            url = 'http://api.k780.com/?app=weather.history&cityId=' + city_id + '&date=' + day \
                  + '&appkey=64688&sign=365edee4d6e1241a6f65ebe2f088f861&format=json'
            raw_data = pd.read_json(url)

            # getting data of 24 hours in one day
            for i in range(len(raw_data)):
                one_hour_data = pd.Series(raw_data.iloc[i, 1])
                # setting time as index and temperature as values
                new_data[one_hour_data['uptime']] = int(one_hour_data['temp'])

        print('online data loaded')
        return pd.DataFrame(new_data, columns=['temperature'])


def window_extraction(raw_data, length, overlap):
    # extracting features in windows
    mean_data = raw_data.rolling(window=length).mean().dropna().reset_index(drop=True)
    max_data = raw_data.rolling(window=length).max().dropna().reset_index(drop=True)
    min_data = raw_data.rolling(window=length).min().dropna().reset_index(drop=True)
    median_data = raw_data.rolling(window=length).median().dropna().reset_index(drop=True)
    std_data = raw_data.rolling(window=length).std().dropna().reset_index(drop=True)

    frame_data = pd.concat([mean_data, max_data, min_data, median_data, std_data], axis=1)
    frame_data.columns = ['mean', 'max', 'min', 'median_data', 'std']

    # reducing the overlap
    new_data = pd.DataFrame()
    j = 0
    while j < frame_data.shape[0]:
        new_data = pd.concat([new_data, frame_data.iloc[j].to_frame().T], ignore_index=True)
        j = j + (length - round(length * overlap))

    return new_data


def train_test(test_ratio):
    # messing the sequence
    i = np.random.permutation(len(mass_normalized_data))

    # defining train and test data
    x = mass_normalized_data
    y = data_class
    test_rate_index = int(-len(mass_normalized_data) * test_ratio)

    x_train = x.iloc[i[:test_rate_index], :]
    y_train = y[i[:test_rate_index]]
    x_test = x.iloc[i[test_rate_index:], :]
    y_test = y[i[test_rate_index:]]

    # training
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    # predicting
    result = knn.predict(x_test)
    accuracy1 = (result == y_test).sum() / len(result)

    print('The accuracy of classification is', accuracy1, ', with a test ratio of', test_ratio, )
    return accuracy1


########################################################################################################################
# Creating dataset
########################################################################################################################

# defining the  time range of the dataset
date = pd.date_range(start='2022-2-23', periods=15, freq='D')

# defining the information of cities
city_ids = ['101310201', '101130101', '101050703']
city_names = ['Sanya', 'Wulumuqi', 'Mohe']

# fetching data
Sanya = get_data(date, city_ids[0], city_names[0])
Wulumuqi = get_data(date, city_ids[1], city_names[1])
Mohe = get_data(date, city_ids[2], city_names[2])

# there is an abnormal value "8" of temperature,
# between "24" and "24" within three hours.
Sanya.iloc[-76] = 24

# saving data locally
Sanya.to_pickle(city_names[0])
Wulumuqi.to_pickle(city_names[1])
Mohe.to_pickle(city_names[2])

# merging all the data
mass_data = pd.concat([Sanya, Wulumuqi, Mohe])

########################################################################################################################
# Description of the dataset
########################################################################################################################

print('The variables are temperature and time')
print('The number of instance is', mass_data.shape[0])
print('The number of classes is 3, they are 3 cities: Sanya, Wulumuqi, Mohe')
print('The minimum of temperature is', mass_data.values.min())
print('The maximum of temperature is', mass_data.values.max())
print('The minimum of time is', mass_data.index.min())
print('The maximum of time is', mass_data.index.max())
print('The head of the data\n', mass_data.head())

########################################################################################################################
# Extracting Features
########################################################################################################################

"""""""""""""""""""""""""""""""""""""""
User parameters of time windows
"""""""""""""""""""""""""""""""""""""""
# defining the sliding time window's length (by number of instances)
window_length = 10
# defining the overlap ratio between two successive windows (by number of instances)
overlap_ratio = 0.8
""""""""""""""""""""""""""""""""""""""""""""

# extracting data
Sanya_extracted = window_extraction(Sanya, window_length, overlap_ratio)
Wulumuqi_extracted = window_extraction(Wulumuqi, window_length, overlap_ratio)
Mohe_extracted = window_extraction(Mohe, window_length, overlap_ratio)

# storing the class of data
data_class = np.array([[0] * len(Sanya_extracted)])
data_class = np.append(data_class, [1] * len(Wulumuqi_extracted))
data_class = np.append(data_class, [2] * len(Mohe_extracted))

# merging all the extracted data
mass_extracted_data = pd.concat([Sanya_extracted, Wulumuqi_extracted, Mohe_extracted])
print('The head of the feature-extracted data\n', mass_extracted_data.head())

# saving the data
mass_extracted_data.to_pickle('feature_data.pickle')

########################################################################################################################
# Inspection and pre-processing
########################################################################################################################

print('The number of missing values', mass_data.isna().sum().values)

# normalization separately
Sanya_normalized = pd.DataFrame(preprocessing.scale(Sanya))
Wulumuqi_normalized = pd.DataFrame(preprocessing.scale(Wulumuqi))
Mohe_normalized = pd.DataFrame(preprocessing.scale(Mohe))

# normalization jointly
mass_normalized_data = pd.DataFrame(preprocessing.scale(mass_extracted_data))

print('Instances number of each classes after pre-processing is')
print(Sanya_normalized.shape[0])
print(Wulumuqi_normalized.shape[0])
print(Mohe_normalized.shape[0])

# drawing a line chart
ax2 = Sanya.plot(color='b')
Wulumuqi.plot(ax=ax2, color='r')
Mohe.plot(ax=ax2, color='y')
plt.legend(city_names)
plt.xlabel('time')
plt.ylabel('temperature')
plt.xlim([0, len(Sanya)])
plt.show()

# drawing a normalized line chart
class_data = pd.concat([Sanya_normalized, Wulumuqi_normalized, Mohe_normalized], axis=1)
class_data.plot()
plt.legend(city_names)
plt.xlabel('normalized time')
plt.ylabel('normalized temperature')
plt.xlim([0, len(Sanya)])
plt.show()

# computing statistics
print('Statistics\n', mass_data.describe())

########################################################################################################################
# Dimensionality reduction
########################################################################################################################

# loading previously saved data
loaded_data = pd.read_pickle('feature_data.pickle')

# applying dimensionality reduction jointly
mass_reduced_data = pd.DataFrame(PCA(n_components=2).fit_transform(mass_normalized_data))

# drawing scatter plot
plt.figure()
plt.scatter(x=mass_reduced_data.iloc[:, 0], y=mass_reduced_data.iloc[:, 1], c=data_class)
plt.title('window length=' + str(window_length) + ', ' + 'overlap ratio=' + str(overlap_ratio))
plt.show()
########################################################################################################################
# Classification
########################################################################################################################

print('The current window length is', window_length, ', overlap ratio is', overlap_ratio)

# defining train and test data
x = mass_normalized_data
y = data_class

# defining the ratio of test data to total data
test_ratio_list = [0.6, 0.5, 0.4, 0.3]

# messing the sequence
i = np.random.permutation(len(mass_normalized_data))

# training and testing
for test_ratio in test_ratio_list:
    test_rate_index = int(-len(mass_normalized_data) * test_ratio)

    x_train = x.iloc[i[:test_rate_index], :]
    y_train = y[i[:test_rate_index]]
    x_test = x.iloc[i[test_rate_index:], :]
    y_test = y[i[test_rate_index:]]

    # training
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    # predicting
    result = knn.predict(x_test)
    accuracy = (result == y_test).sum() / len(result)

    print('The accuracy of classification is', accuracy,
          ', with a train/test ratio of', int((1 - test_ratio) * 100), '/', int(test_ratio * 100))

# 10-fold cross-validation

accuracy_sum = 0
i_list = np.array_split(i, 10)
for num in range(10):
    i_nine = np.array(np.array_split(i, 10).pop(num)).ravel()
    x_train = x.iloc[i_nine, :]
    y_train = y[i_nine]
    x_test = x.iloc[np.array_split(i, 10)[num], :]
    y_test = y[np.array_split(i, 10)[num]]
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    result = knn.predict(x_test)
    accuracy = (result == y_test).sum() / len(result)
    accuracy_sum += accuracy

print('The 10-fold cross-validation result is', accuracy_sum / 10)

# drawing decision boundary in scatter plot
x = mass_normalized_data.iloc[:, :2]
y = data_class

x_min, x_max = x.iloc[:, 0].min() - .5, x.iloc[:, 0].max() + .5
y_min, y_max = x.iloc[:, 1].min() - .5, x.iloc[:, 0].max() + .5

cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
h = .02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
knn = KNeighborsClassifier()
knn.fit(x, y)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
