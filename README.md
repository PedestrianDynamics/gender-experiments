# Gender-experiments
[![gender-experiments-world](https://github.com/PedestrianDynamics/gender-experiments/actions/workflows/code-quality.yml/badge.svg)](https://github.com/PedestrianDynamics/gender-experiments/actions/workflows/code-quality.yml)
[![Open Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gender-experiments.streamlit.app/)

The file naming convention is structured as follows:


```bash
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



##  RUNNING 

You can run the hosted version of this app by clicking the streamlit badge above.
Alternatively, you can also run it locally:

1. Install requirements

```bash
pip installl -r requirements
```

2. Run the app

```
streamlit run app.py
```

## ACKNOWLEDGMENT

We thank all the colleagues who helped with the organization of the experiments and extraction of the trajectories, as well as the curation of the data. Special thanks go to Alica Kandler for the curation and data quality of the German data, Reza Shahbad for managing the logistics of the experiments conducted in Australia, and Maziar Yazdani for the curation of the Australian data and Shi Dongdong (experiments in China).

Claudio Feliciani and Xiaolu Jia express their gratitude for the funding received through the JST-Mirai Program grant numbers JPMJMI20D1 and the JSPS KAKENHI grant numbers JP20K14992, JP23K13521, and JP21K14377. They also appreciate the support from www.jikken-baito.com and the Meguro Senior Employment Center in recruiting participants.

Milad Haghani acknowledges the support from the Australian Research Council, with grant number DE210100440.

Jian Ma acknowledges funding provided by the National Natural Science Foundation of China (Nos. 72104205) and the National Key Research and Development Program of China (No. 2022YFC3005205).

The German experiment, detailed at https://doi.org/10.34735/ped.2021.2, was part of an experimental series for the CroMa and CrowdDNA projects. These experiments took place at the Mitsubishi Electric Halle in Düsseldorf, Germany, in 2021.
