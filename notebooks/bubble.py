from wordcloud import WordCloud
import matplotlib.pyplot as plt

# List of terms
terms = [
    "Mikroskopische Räumungsanalyse", "Räumung", "Entfluchtung", "Personenverteilung",
    "Populationseigenschaften", "Staus", "Freie Laufgeschwindigkeit", "Detektionszeit",
    "Alarmierungszeit", "Individuelle Reaktionszeit", "Individuelle Laufzeit",
    "Individuelle Räumungszeit", "Räumungszeit", "Agent", "Szenario",
    "Statistische Auswertung wiederholter Simulationsläufe"
]

# Convert the list to a single string
text = ' '.join(terms)

# Create the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
