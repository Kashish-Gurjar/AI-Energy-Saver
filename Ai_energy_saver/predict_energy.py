import pickle 
import pyttsx3
import pandas as pd  # ✅ Add pandas for DataFrame support

# Load trained models
clf = pickle.load(open("clf.pkl", "rb"))
reg = pickle.load(open("reg.pkl", "rb"))

# Mapping for categorical data
device_map = {'Fan': 0, 'Light': 1, 'AC': 2}
day_map = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

# Text-to-speech function
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Predict function (now with DataFrame input)
def predict(device, hour, day):
    # ✅ Create DataFrame with proper column names
    input_data = pd.DataFrame([[device_map[device], hour, day_map[day]]],
                              columns=["device", "hour", "day"])
    
    # Predict using classifier and regressor
    should_on = clf.predict(input_data)[0]
    duration = reg.predict(input_data)[0]

    # Output and speak the result
    msg = f"{device} may run for {int(duration)} mins. Suggestion: {'Keep it ON' if should_on else 'Turn it OFF'}"
    print(msg)
    speak(msg)

# Example call
predict("Fan", 10, "Tue")
