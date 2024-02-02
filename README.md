# Gender-experiments
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gender-experiments.streamlit.app/)


## FILENAME LOGIC ####

Name of the files is:
`
country/type_people_repetition
`

where,

- country:		aus, chn, ger, jap, pal
- type:			gender composition (female, male, mix_random, mix_sorted)
- people:			number of people in the course (the actual number)
- repetition:		number indicating the repetition for a specific condition


## CSV FILE LOGIC ####
Columns of the files are:
`
ID,gender,frame,t(s),x(m),y(m)
`

| Field    | Description                              |
|----------|------------------------------------------|
| ID       | id of participant (pedestrian)           |
| next     | id of next neighbor (-1 is no neighbor)                      |
| prev     | id of previous neighbor (-1 is no neighbor)                  |
| gender   | 1 = female, 2 = male                     |
| frame    | frame number as integer                  |
| t(s)     | time in seconds                          |
| x(m)     | x coordinate (m)                         |
| y(m)     | y coordinate (m)                         |
