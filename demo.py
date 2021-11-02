"""
Social Media Analytics Project
Name:
Roll Number:
"""
import re

from nltk.util import clean_url
import hw6_social_tests as test

project = "Social" # don't edit this

### PART 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    data = pd.read_csv(filename)
    return data


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    for line in fromString.split("\n"):
        length = line.find(" ") 
        string1 = line[length:]
        bracket = string1.find( "(" )
        string2 = string1[:bracket]
        line = string2.strip()
    return line
    


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    for line in fromString.split("\n"):
        length = line.find(" (") + \
        len(" (")
        # leng = length+1
        string1 = line[length:]
        from_ = string1.find(" from")
        string2 = string1[:from_].strip()
    return string2
    


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    for line in fromString.split("\n"):
        start = line.find(" from ") + \
        len(" from ")
        line = line[start:]
        end = line.find(")")
        line = line[:end].strip()
    return line


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):            
    # return multiple
    # res = re.findall(r'\#\w+|endchar\$', message)
    lst=[]
    m = message.split("#")
    for x in m[1:len(m)]: 
        string=""
        for y in x:
            if y not in endChars:
                string+=y
            else:
                break
        string="#"+string
        lst.append(string)
       
    return lst


'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    result = stateDf.loc[stateDf["state"] == state, "region"]
    return result.values[0]


'''
addColumns(data, stateDf)
#7 [Check6-1]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    name_add=[]
    position_add = []
    state_add=[]
    region_add=[]
    hastag_add=[]
    for index,row in data.iterrows():
        column_value = data['label'].iloc[index]
        name = parseName(column_value)
        position = parsePosition(column_value)
        state = parseState(column_value)
        region = getRegionFromState(stateDf, state)
        text_value = data['text'].iloc[index]
        hastag = findHashtags(text_value)
        name_add.append(name)
        position_add.append(position)
        state_add.append(state)
        region_add.append(region)
        hastag_add.append(hastag)
    data['name']=name_add
    data['position']=position_add
    data['state'] = state_add
    data['region'] = region_add
    data['hashtags'] = hastag_add
    print(len(data['hashtags']))
    return None

# df = makeDataFrame("data/politicaldata.csv")
# stateDf = makeDataFrame("data/statemappings.csv")
# print(addColumns(df, stateDf))

### PART 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    elif score > 0.1:
        return "positive"
    return "neutral"


'''
addSentimentColumn(data)
#2 [Check6-2]
Parameters: dataframe
Returns: None
'''
def addSentimentColumn(data):
    classifier = SentimentIntensityAnalyzer()
    sentiments=[]
    for index,row in data.iterrows():
        message = data['text'].iloc[index]
        result = findSentiment(classifier, message)
        sentiments.append(result)
    data['sentiment']=sentiments
    return None


'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''

def getDataCountByState(data,colName, dataToCount):
    newd={}
    # print("datacolname:",data[colName])
    if len(colName) !=0 and  len(dataToCount) !=0:
        for index,row in data.iterrows():
            if row[colName] == dataToCount:
                if row['state'] not in newd:
                    print("state:", row['state'])
                    newd[row['state']]=0
                newd[row['state']]+=1
    elif colName=="" and dataToCount=="":
        for index,row in data.iterrows():
            if row['state'] not in newd:
                newd[row['state']]=0
            newd[row['state']]+=1
    return newd 


# df = makeDataFrame("data/politicaldata.csv")
# stateDf = makeDataFrame("data/statemappings.csv")
# addColumns(df, stateDf)
# addSentimentColumn(df)
# print(getDataCountByState(df, "message", "attack"))



'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    dict_={}
    for index,row in data.iterrows():
        if row['region'] not in dict_:
            dict_[row['region']]={}
        if row[colName] not in dict_[row['region']]:
            dict_[row['region']][row[colName]]=0
        dict_[row['region']][row[colName]]+=1
    return dict_


'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
'''
notice:
we get hasable type error when you are trying to assign list as key to the dict

'''
def getHashtagRates(data):
    result_dict = {}
    # print("tags:",data['hashtags'])
    for key in data["hashtags"]:
        for Dkey in key:
            if len(Dkey)!=0 and Dkey not in result_dict:
                result_dict[Dkey]=1
            else:
                result_dict[Dkey]+=1
    # print(len(result_dict))
    # print(result_dict)
    return result_dict 
    
# df = makeDataFrame("data/politicaldata.csv")
# stateDf = makeDataFrame("data/statemappings.csv")
# addColumns(df, stateDf)
# addSentimentColumn(df)
# print(getHashtagRates(df))


'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
import operator
def mostCommonHashtags(hashtags, count):
    result={}
    sorted_hastags = dict( sorted(hashtags.items(), key=operator.itemgetter(1),reverse=True))
    for i in range(count):
        for x,y in sorted_hastags.items():
            if len(result)!=count:
                result[x]=y
    return result
    


'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
def getHashtagSentiment(data, hashtag):
    count=0
    lst=[]
    for index,row in data.iterrows():
        if hashtag in row['text']:
            count+=1
            if row['sentiment'] == 'positive':
                lst.append(1)
            elif row['sentiment'] == 'negative':
                lst.append(-1)
            elif row['sentiment'] == 'neutral':
                lst.append(0)
    # print(count)
   

    return sum(lst)/count


### PART 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    keys=[]
    values=[]
    for x,y in stateCounts.items():
        keys.append(x)
        values.append(y)
    # plt.xticks(values, keys,  rotation="vertical")
    plt.xticks(ticks=list(range(len(values))), labels=keys, rotation="vertical")
    plt.title(title)
    plt.bar(keys, values)
    plt.show()
    # print("keys:", keys)
    return


'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
from collections import Counter
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    featurerate = {} 
    topstates = {} 
    for i in stateFeatureCounts: 
    #    print(i)
    #    print("avg",stateFeatureCounts[i] / stateCounts[i]) 
       featurerate[i] = (stateFeatureCounts[i] / stateCounts[i]) 
    #    print(featurerate)
       topstates = dict(Counter(featurerate).most_common(n)) 
    graphStateCounts(topstates, title)
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    featuring_list=[]
    region_list=[]
    region_values=[]
    for item in regionDicts:
        temp_list=[]
        for x in regionDicts[item]:
            if x not in featuring_list:
                featuring_list.append(x)
            temp_list.append(regionDicts[item][x])
        region_values.append(temp_list)
        region_list.append(item)
    # sideBySideBarPlots(featuring_list,region_list,region_values, title)
    # print("regiondict", regionDicts)


    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    hashtag_list=[]
    frequency_list=[]
    sentiment_score=[]
    rate = getHashtagRates(data)
    tag = mostCommonHashtags(rate,50)
    for x,y in tag.items():
        hashtag_list.append(x)
        frequency_list.append(y)
        sentiment = getHashtagSentiment(data,x)
        sentiment_score.append(sentiment)
    # string = ' '.join(map(str, hashtag_list)) 
    print("f",frequency_list)
    scatterPlot(frequency_list,sentiment_score, hashtag_list,"SentimentByFrequency")
    return


#### PART 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

if __name__ == "__main__":
    # print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    # test.week1Tests()
    # print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    # test.runWeek1()

    ## Uncomment these for Week 2 ##
    """print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()"""

    ## Uncomment these for Week 3 ##
    """print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()"""
    #test.testMakeDataFrame()
    #test.testParseName()
    #test.testParsePosition()
    #test.testParseState()
    #test.testFindHashtags()
    #test.testGetRegionFromState()
    #test.testAddColumns()
    #test.week1Tests()
    #test.testFindSentiment()
    #test.testAddSentimentColumn()
    test.runWeek3()