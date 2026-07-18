# Mobile Product Segmentation and Recommendation System Using Python and Machine Learning

## Project Overview
This project is a Streamlit web application for mobile product segmentation and recommendation. It helps users explore how mobile products are grouped into clusters and get personalized recommendations based on a few product preferences such as budget, rating, battery life, camera quality, performance, design, and display.

The app combines clustering insights with a content-based similarity recommendation approach. Instead of selecting a product by index, users enter a desired profile and the system ranks matching mobiles by similarity.

## Key Features
- Feature-based mobile recommendations using cosine similarity.
- Cluster analysis with visual explanations of product segments.
- Clean Streamlit interface with sliders and filters.
- Product name display instead of raw row index.
- Visual analytics for cluster size, price vs rating, feature heatmap, and radar profiles.
- Support for brand preference filtering.

## How It Works
1. The dataset is loaded from the `Dataset` folder.
2. Product features are normalized so they can be compared fairly.
3. The user fills in a mobile preference profile in the sidebar.
4. The app computes cosine similarity between the selected profile and the available products.
5. The most similar mobiles are ranked and displayed as recommendations.
6. Cluster charts summarize how the dataset is segmented across the three discovered clusters.

## Recommendation Logic
This application uses a content-based recommendation approach.

- It does not rely on collaborative filtering or user-user interaction history.
- It compares feature values such as price, ratings, battery life, camera score, performance score, design score, and display score.
- It also allows optional brand filtering.

## Dataset
The project uses the dataset stored in the `Dataset` folder. In the current workspace, the main file is:

- `Dataset/segmented_mobile_products.xls`

Important note:
- The file has an `.xls` extension, but its contents are CSV text.
- The app handles this automatically.

Main columns used by the app include:
- `price_usd`
- `rating`
- `battery_life_rating`
- `camera_rating`
- `performance_rating`
- `design_rating`
- `display_rating`
- `helpful_votes`
- `Cluster`

The dataset also contains one-hot encoded columns for mobile names, brands, sentiments, regions, and sources.

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn

## Installation
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the App
Start the Streamlit application with:

```bash
streamlit run app.py
```

## How to Use
1. Open the app in your browser.
2. Adjust the sliders in the sidebar to describe your preferred mobile.
3. Optionally choose a brand preference.
4. Click the recommendation button.
5. Review the recommended phones, similarity score, and cluster insights.

## Project Structure
```text
Mobile Product Segmentation and Recommendation System Using Python and Machine Learning/
├── app.py
├── requirements.txt
├── Dataset/
│   ├── segmented_mobile_products.xls
│   ├── Processed_data.xls
│   ├── Scaled_data.xls
│   └── Mobile Reviews Sentiment null.csv
└── Notebook/
	├── Data Cleaning.ipynb
	└── Model.ipynb
```

## Visualizations Included
- Cluster size distribution
- Price vs rating scatter plot
- Average feature heatmap by cluster
- Radar chart for cluster profiles

## Project Insights
The clusters help separate the dataset into different market segments. Based on the current labels used in the app, the clusters represent:

- Premium underperformers
- Crowd pleasers
- Budget traps

These labels make it easier to understand the product groups and explain why some phones are recommended over others.

## Future Improvements
- Add a true trained recommendation model.
- Allow users to compare phones side by side.
- Add search and sort options by brand or price.
- Store user feedback to improve future recommendations.
- Convert the mislabeled dataset file into a proper `.csv` or `.xlsx` file for cleaner maintenance.

## Author
Puneeth Raja