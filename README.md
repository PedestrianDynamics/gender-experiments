# Gender-experiments
[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gender-experiments.streamlit.app/)

The file naming convention is structured as follows:

bash
```
country/type_people_repetition
```

Here's what each component represents:

- `country`: This signifies the country code, which could be one of the following: aus, chn, ger, jap, pal.
- `type`: Indicates the type of composition, focusing on the gender mix within the data. 
   Options include 
   - female, 
   - male, 
   - mix_random, and 
   - mix_sorted.
- `people`: Reflects the total number of individuals within the course, represented by the actual count.
- `repetition`: This is a numerical value that signifies the repetition number for a particular condition.

## CSV FILE FORMAT

The files adhere to the following column structure:


```bash
ID,gender,frame,t(s),x(m),y(m)
```

The columns are defined as follows:

| Field    | Description                              |
|----------|------------------------------------------|
| ID       | An identifier for each entry.           |
| next     | id of next neighbor (-1 is no neighbor)                      |
| prev     | id of previous neighbor (-1 is no neighbor)                  |
| gender   | Denotes the gender of the individual. 1 = female, 2 = male   |
| frame    | Represents the frame number in the data.                 |
| t(s)     | Indicates the time in seconds.                          |
| x(m)     | Specifies the X-coordinate in meters.                         |
| y(m)     | Specifies the Y-coordinate in meters.                        |


## ACKNOWLEDGMENT

Acknowledgments are greatly appreciated. 
Feel free to include a sentence in the README.md file to acknowledge any individuals or entities you wish to recognize, and consider mentioning a project number, especially if funding is involved.


