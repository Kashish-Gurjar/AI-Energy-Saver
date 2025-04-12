import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Generate fake device usage data
devices = ['Fan', 'Light', 'AC']
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
data = []

for _ in range(1000):
    device = random.choice(devices)
    hour = random.randint(0, 23)
    day = random.choice(days)
    duration = random.randint(5, 180)
    data.append([device, hour, day, duration])

df = pd.DataFrame(data, columns=['device', 'hour', 'day', 'duration'])

# Encode categorical data
df['device'] = df['device'].astype('category').cat.codes
df['day'] = df['day'].astype('category').cat.codes

# Features and targets
Fan = df[['device', 'hour', 'day']]
y_class = (df['duration'] > 60).astype(int)
y_reg = df['duration']

# Train models
clf = RandomForestClassifier().fit(X, y_class)
reg = RandomForestRegressor().fit(X, y_reg)

# Save models
pickle.dump(clf, open("clf.pkl", "wb"))
pickle.dump(reg, open("reg.pkl", "wb"))

print("✅ Model training complete!")
