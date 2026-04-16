from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def process_data(data):

    data['Engagement'] = data['Likes_Per_Day'] / (data['Posts_Per_Day'] + 1)

    data['Activity_Score'] = (
        data['Daily_Minutes_Spent'] +
        data['Posts_Per_Day'] * 10 +
        data['Follows_Per_Day'] * 2
    )

    features = [
        'Daily_Minutes_Spent',
        'Posts_Per_Day',
        'Likes_Per_Day',
        'Follows_Per_Day',
        'Engagement',
        'Activity_Score'
    ]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_data)

    def label_user(row):
        if row['Activity_Score'] > data['Activity_Score'].mean():
            return "Highly Active"
        elif row['Engagement'] > data['Engagement'].mean():
            return "Engaged"
        else:
            return "Low Activity"

    data['User_Type'] = data.apply(label_user, axis=1)

    return data
