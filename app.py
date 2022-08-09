import streamlit as st
import preproceesor,helper
import matplotlib.pyplot as plt
import seaborn as sns


st.sidebar.title("CHAT ANALYZER")

uploaded_file=st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data=uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    # st.text(data)
    df=preproceesor.preprocess(data)

    #for printing dataset of chat
    # st.dataframe(df)

    #fetch unique users
    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user=st.sidebar.selectbox("Show analysis wrt",user_list )

    if st.sidebar.button("Show Analysis"):

        #Stats Area
        num_messages,words,num_media_messages,links=helper.fetch_stats(selected_user,df)

        st.title("Top Statistics")
        col1,col2,col3,col4=st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Media Messages")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(links)


        #monthly_timeline

        st.title("Monthly_Timeline")
        timeline=helper.monthly_timeline(selected_user, df)

        fig,ax=plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #daily_timeline
        st.title("Daily_Timeline")
        dly_timeline=helper.daily_timeline(selected_user, df)

        fig,ax=plt.subplots()
        ax.plot(dly_timeline['only_date'], dly_timeline['message'],color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #activity map
        st.title("Activity Map")
        col1,col2=st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day=helper.week_activity_map(selected_user,df)

            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        st.title("Weekly Activity Map")
        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)


        # finding the busiest users in the group
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()

            col1,col2=st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


        #wordcloud
        st.title('Word Cloud')
        df_wc=helper.create_wordcloud(selected_user, df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most_common_words

        st.title('Most Common Words')
        most_common_df=helper.most_common_words(selected_user,df)
        #
        # col1, col2 = st.columns(2)
        #
        # with col1:
        #     st.dataframe(most_common_df)
        # with col2:

        fig,ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # wordcloud without stopwords
        st.title('Word Cloud without Stopwords')
        df_wc_s = helper.create_wordcloud_without_stopword(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc_s)
        st.pyplot(fig)

        # emoji_analysis

        emoji_df=helper.emoji_helper(selected_user,df)
        st.title("Emoji Analyzer")

        col1,col2=st.columns(2)

        with col1:
            # st.figure(figsize=(18,10))
            st.dataframe(emoji_df)

        with col2:
            fig,ax=plt.subplots()
            # ax.bar(emoji_df[0].head(),emoji_df[1].head())
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)

        #sentiment analysis

        sentiment_df = helper.sentiment_analyze(selected_user, df)
        st.title("Sentiment Analyzer")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(sentiment_df)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(sentiment_df[0].head(),sentiment_df[1].head())
            st.pyplot(fig)

        #alphabet

        st.title("Alphabet Analyzer")
        alphabet_df=helper.alphabet_analyze(selected_user,df)

        col1,col2=st.columns(2)

        with col1:
            fig,ax=plt.subplots()
            ax.bar(alphabet_df[0],alphabet_df[1])
            st.pyplot(fig)
        with col2:
            st.dataframe(alphabet_df)





