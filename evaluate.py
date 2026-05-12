import requests
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, cohen_kappa_score)
import numpy as np

reviews = [
    {"text": "Absolutely fantastic experience! Best place in Kathmandu by far. Highly recommend to everyone. Will definitely come back again soon!", "rating": 5},
    {"text": "Very bad place to swim. Small pool very crowded and it is only for their swim classes. Do not go here.", "rating": 1},
    {"text": "We enjoyed this hotel very much. The staff works really hard to make you feel at home. What a collection of buildings. Impressive. Do not miss it.", "rating": 5},
    {"text": "Terrible service, waited 45 minutes for food that was cold when it arrived. Staff were rude and unapologetic. Never coming back.", "rating": 1},
    {"text": "Great mall, great meetup place and many stalls", "rating": 4},
    {"text": "This is the best restaurant I have ever visited in my entire life. The food quality is outstanding and service is top notch.", "rating": 5},
    {"text": "The worst experience of local kukhara thakali till date. The meat was not cooked properly.", "rating": 2},
    {"text": "Food was delicious and staffs were friendly. I really enjoyed my visit there.", "rating": 5},
    {"text": "Exceptional quality and service. I have visited many places but this stands out as truly remarkable. Strongly recommended.", "rating": 5},
    {"text": "Does not give ice cream to male people (mostly children)", "rating": 1},
    {"text": "Staying at Hotel Yak & Yeti was a wonderful experience from the moment I arrived. The staff greeted us warmly, making us feel instantly welcome. Throughout my stay, they were friendly, professional, and always willing to help.", "rating": 5},
    {"text": "Overpriced and underwhelming. Expected much better given the reputation. Food was bland and portions were tiny.", "rating": 2},
    {"text": "The mall is clean and well maintained. Plenty of shopping venues to view with memoriable and fun things to do like archery.", "rating": 4},
    {"text": "Perfect in every way. Clean, professional, friendly staff. Food was incredible. Best value for money in Kathmandu.", "rating": 5},
    {"text": "Although the food was good but they took about 1 and half hour just to prepare momo", "rating": 2},
    {"text": "One of the best in current malls in Nepal in current date. Clean, good atmosphere with lots of options from shopping to entertainment", "rating": 4},
    {"text": "Not worth the hype. Mediocre food, slow service, and the place was not as clean as expected. Save your money.", "rating": 2},
    {"text": "Great service superb food and the ambience was spectacular we'll be back and she said she'll marry me!", "rating": 5},
    {"text": "Excellent place for working out. Got everything you need and more", "rating": 5},
    {"text": "Had high expectations but left feeling let down. The quality has clearly dropped. Management needs to address these issues urgently.", "rating": 1},
    {"text": "Very organised and impressed with the staffs and the management. The venue itself looks elegant and pretty. Will surely be looking forward for other events too.", "rating": 5},
    {"text": "Amazing staff, amazing food, amazing ambience. Everything was just perfect. 10/10 would recommend to all my friends and family.", "rating": 5},
    {"text": "Waste of money ...Worst service and I will not recommend anyone ...", "rating": 2},
    {"text": "I visited last week and had an unforgettable experience. The ambience was lovely and the staff went above and beyond.", "rating": 5},
    {"text": "Recently visited this banquet. All the interior is so good but toilet doesn't match with the banquet at all. I almost vomited. dirty, small it felt like I went to a public toilet. My party Gown was ruined here.", "rating": 5},
    {"text": "Lost 1 star due to food. Service is nice", "rating": 4},
    {"text": "Superb experience from start to finish. Every detail was perfect. The team here truly cares about customer satisfaction.", "rating": 5},
    {"text": "Extremely noisy! Use Soundproofing measures and limit the noise without the boundary", "rating": 1},
    {"text": "Great fitness club wide range of service, nice pool with great coaching sessions, food also is good", "rating": 5},
    {"text": "Disappointed with my visit. The place looked nothing like the photos. Staff seemed disinterested and unhelpful.", "rating": 2},
    {"text": "It was nice time here, very clean and quiet.", "rating": 4},
    {"text": "Wonderful place! Great experience overall. The staff were very helpful and the food was delicious. Must visit!", "rating": 5},
    {"text": "The food was absolutely delicious. The flavors were well balanced and everything tasted fresh.", "rating": 5},
    {"text": "I liked the Banquet esp its decoration and tasty food. Suitable for party function for around 300 people.", "rating": 5},
    {"text": "Had a 2-day stay at the hotel. Rooms are okay-ish. Internet/wifi management is poor and unreliable. Bike parking area is horrible.", "rating": 1},
    {"text": "Nice mall with good shops and atmosphere", "rating": 4},
    {"text": "One of the finest establishments in Nepal. Impeccable service, outstanding food, beautiful interior. Highly recommended!", "rating": 5},
    {"text": "Food and service were good.. but the place is a bit old as compared to new opening party palaces", "rating": 3},
    {"text": "The Venue is Spacious. But no proper Parking, you have to park at another venue and walk 8/10 mins to reach the venue. Not adequate ventilation and Cooling. The food was also subpar.", "rating": 2},
    {"text": "Lovely place, great food and warm hospitality. Enjoyed every moment of our visit. Will definitely return!", "rating": 5},
    {"text": "Very nice, beautiful and quiet place to stay", "rating": 5},
    {"text": "Great place with excellent facilities. Staff is courteous and professional. Food is delicious. Will visit again for sure.", "rating": 5},
    {"text": "Disappointing Experience. I recently stayed at Hotel Yak & Yeti and was extremely disappointed by the way the situation I faced was handled by both the staff and management.", "rating": 2},
    {"text": "Staff need to be more attentive towards their guests although internal problems persist sometimes. Good luck.", "rating": 5},
    {"text": "Really impressed with the quality and attention to detail. Staff were wonderful and made us feel very special.", "rating": 5},
    {"text": "Outstanding, very special location. We went for dinner and it couldn't be better: service, food atmosphere were very unique.", "rating": 5},
    {"text": "Fantastic experience overall. Clean, comfortable and great value. Highly recommend to anyone visiting Kathmandu.", "rating": 5},
    {"text": "Prime location, near by Jhamsikhel, Very nice view, can see and by branded products.", "rating": 5},
    {"text": "Thoroughly enjoyed our time here. Excellent food, friendly staff and a wonderful atmosphere. Can't wait to come back.", "rating": 4},
    {"text": "Beautiful setting with outstanding service. The food was fresh and flavorful. One of the best experiences in the city.", "rating": 5},
]

# 1=real, 0=fake for all annotators
A1 = [0,1,0,1,0,0,1,1,0,1,0,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,1,0,0,1,0,0]
A2 = [0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1]
A3 = [1,1,0,1,1,0,1,0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,1]
A4 = [0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,1,1]

# majority vote: >=3 out of 4
majority = []
for i in range(50):
    votes = A1[i] + A2[i] + A3[i] + A4[i]
    majority.append(1 if votes >= 3 else 0)

print(f"Majority Real(1): {majority.count(1)}, Fake(0): {majority.count(0)}")

# pairwise kappa
pairs = [('A1','A2',A1,A2),('A1','A3',A1,A3),('A1','A4',A1,A4),
         ('A2','A3',A2,A3),('A2','A4',A2,A4),('A3','A4',A3,A4)]

print("\n── Inter-Annotator Agreement ──")
kappas = []
for n1, n2, x, y in pairs:
    k = cohen_kappa_score(x, y)
    kappas.append(k)
    agree = sum(a == b for a, b in zip(x, y))
    print(f"  {n1} vs {n2}: kappa={k:.3f}, agreement={agree}/50 ({agree*2}%)")
print(f"  Average kappa: {np.mean(kappas):.3f}")

# get model predictions
response = requests.post("http://127.0.0.1:5000/analyze", json={"reviews": reviews})
data = response.json()

model_preds = []
for r in data["reviews"]:
    if r["label"] == "Real":
        model_preds.append(1)
    else:  # Fake or Suspicious
        model_preds.append(0)

print(f"\nModel Real(1): {model_preds.count(1)}, Fake(0): {model_preds.count(0)}")

acc  = accuracy_score(majority, model_preds)
f1   = f1_score(majority, model_preds, zero_division=0)
prec = precision_score(majority, model_preds, zero_division=0)
rec  = recall_score(majority, model_preds, zero_division=0)
cm   = confusion_matrix(majority, model_preds)

print(f"\n── Model vs Human Majority ──")
print(f"Accuracy:  {acc:.3f}")
print(f"F1:        {f1:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"\nConfusion Matrix (1=real, 0=fake):")
print(f"              Pred Real   Pred Fake")
print(f"Actual Real      {cm[1][1]}          {cm[1][0]}")
print(f"Actual Fake      {cm[0][1]}          {cm[0][0]}")
print(f"\nModel predictions: {model_preds}")
print(f"Majority labels:   {majority}")