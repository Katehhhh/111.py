import pandas as pd
data = pd.read_csv(r'example/soundscape.BirdNET.results.csv',sep=',')
start = data["Start (s)"]
end = data["End (s)"]
CN = data["Common name"]
conf = data["Confidence"]
print(type(conf[0]))