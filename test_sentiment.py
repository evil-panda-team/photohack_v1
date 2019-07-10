import indicoio
indicoio.config.api_key = 'cd048edbe759544dfcd983946d18b1cf'

# single example
#text1 = indicoio.sentiment("It is very bad, I am so sad")

text1 = indicoio.emotion(["Hey, how you doin’?",
"Wanna hang out?",
"Sorry, l’m stuck in traffic jam and will be 30 mins late",
"Are you kidding me?",
"Epic fail!",
"I just don’t love you any more, we need to break up.",
"ROTFL",
"No time to explain, I’m on hackathon",
"I am missing you, come back soon",
"Happy National Astronaut Day!",
"Don’t mess with me!",
"Thank you, but no thank you",
"Buy three packs of tea, 1 kg of sugar and candies"])

print(text1)
