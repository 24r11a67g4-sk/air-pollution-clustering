# Air Pollution Data Analysis Using Clustering

## Objective

This DWDM PBL project analyzes air pollution readings from Indian cities using
unsupervised machine learning. The application preprocesses pollutant
measurements, standardizes the numerical features, and uses K-Means clustering
to group cities with similar pollution profiles. It automatically interprets
the clusters by comparing their average AQI and generates visualizations for
the report.

## Technologies Used

- Python 3
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- HTML5, CSS3, and JavaScript

## Algorithm

K-Means clustering partitions the standardized observations into `k` groups.
Each observation is assigned to the nearest centroid, then the centroids are
updated repeatedly until the assignments stabilize. This application uses
`random_state=42` and `n_init=10` for reproducible clustering.

The available values of `k` are 2, 3, and 4. Cluster numbers are produced by
the algorithm and are not treated as pollution severity. Instead, the app
sorts cluster averages by AQI and assigns Low, Moderate, High, or Severe
Pollution labels based on the selected number of groups.

## Project Structure

```text
air_pollution_clustering/
├── app.py
├── requirements.txt
├── air_pollution.csv
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   ├── script.js
│   └── plots/
└── README.md
```

## Installation

Open the project folder in VS Code and create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The dashboard starts with the included `air_pollution.csv` dataset. Use the
upload panel to analyze another CSV. Uploaded files must contain:

```text
City, PM2.5, PM10, NO2, SO2, CO, AQI
```

The backend handles missing numeric values using the feature mean before
standardization. Every clustering run recreates the elbow, city comparison,
cluster scatter, and cluster distribution plots in `static/plots/`.