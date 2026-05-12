import pickle

with open('model/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

features = tfidf.get_feature_names_out()
features_list = list(features)

interesting = [f for f in features_list if any(word in f for word in [
    'recommend', 'best', 'amazing','worst', 'terrible', 
    'excellent', 'horrible', 'perfect', 'awful',
    'disappoint', 'waste', 'love', 'hate', 'great', 'poor'
])]

print(f"Total features: {len(features)}")
print(f"\nSample of sentiment-rich features learned by the model:")
print("\n".join(interesting[:30]))