from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from textblob import TextBlob

extract=URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df=df[df['user']==selected_user]

    # 1. fetch No. of messages
    num_messages = df.shape[0]

    # 2.fetch the total no. of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # 3.fetch the number of Media messages
    num_media_messages=df[df['message']=='<Media omitted>\n'].shape[0]

    #4. fetch number of links shared
    links=[]
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

    # if selected_user == 'Overall':
    #     #1. fetch No. of messages
    #     num_messages=df.shape[0]
    #
    #     #2.no. of words
    #     words = []
    #     for message in df['message']:
    #         words.extend(message.split())
    #
    #     return num_messages,len(words)
    # else:
    #     new_df=df[df['user']==selected_user]
    #     num_messages=new_df.shape[0]
    #     words = []
    #     for message in new_df['message']:
    #         words.extend(message.split())
    #
    #     return num_messages,len(words)

def most_busy_users(df):
    x=df['user'].value_counts().head()
    df=round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    if selected_user != 'Overall':
        df=df[df['user']==selected_user]

    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    df_wc=wc.generate(df['message'].str.cat(sep=" "))
    return df_wc

def create_wordcloud_without_stopword(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        lst=[]
        for word in message.lower().split():
            if word not in stop_words:
                lst.append(word)
        return " ".join(lst)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message']=temp['message'].apply(remove_stop_words)
    df_wc_s = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc_s

def most_common_words(selected_user,df):

    f=open('stop_hinglish.txt','r')
    stop_words=f.read()

    if selected_user != 'Overall':
        df=df[df['user']==selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words=[]

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df= pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time']=time
    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    dly_timeline=df.groupby('only_date').count()['message'].reset_index()

    return dly_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]


    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]


    return df['month'].value_counts()


def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)


    return user_heatmap

def sentiment_analyze(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiment=[]
    for message in df['message']:
        edu = TextBlob(message)
        x = edu.sentiment.polarity
        if x < 0:
            sentiment.append("Negative")
        elif x == 0:
            sentiment.append("Neutral")
        elif x > 0 and x <= 1:
            sentiment.append("Positive")


    sentiment_df=pd.DataFrame(Counter(sentiment).most_common(len(Counter(sentiment))))

    return sentiment_df

def alphabet_analyze(selected_user,df):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    map = []
    l2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v",
          "w", "x", "y", "z"]
    for j in words:
        for i in j:
            if i in l2:
                map.append(i)


    alphabet_df = pd.DataFrame(Counter(map).most_common(len(map)))
    return alphabet_df