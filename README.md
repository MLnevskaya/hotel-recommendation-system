# hotel-recommendation-system
Content-based hotel recommender using structured features and text embeddings.

## Project Overview

This project builds a hotel recommendation system for hotels in Turkey.
The goal is to recommend similar hotels based on available structured attributes, service information, geographic proximity, hotel category, star rating, and semantic similarity of hotel descriptions.

The final solution combines two approaches:

* **Feature-based similarity** using structured hotel attributes
* **Embedding-based similarity** using canonical hotel text representations

The combined approach makes the recommendation logic more interpretable while still capturing semantic similarity between hotels.

## Dataset

The dataset contains hotel-level information, including:

* hotel ID and name
* geographic coordinates
* hotel category and star rating
* number of rooms
* pet policy
* service groups, such as food services, room amenities, pool, beach, sports, children services, internet, and others

Some fields contain missing or incomplete information, so the project includes additional missing-data handling and comparison reliability checks.

## Project Structure

```text
.
├── 0_preprocessing.ipynb
├── 1_modeling.ipynb
├── 2_inference.ipynb
└── README.md
```

## Notebooks

### 0_preprocessing.ipynb

This notebook prepares the raw hotel data for modeling.

Main steps:

* cleaning hotel categories and star ratings
* processing missing values
* transforming service columns into lists
* creating geographic clusters
* creating `canonical_text` for each hotel
* saving the preprocessed dataset

### 1_modeling.ipynb

This notebook builds similarity representations.

Main steps:

* creating a feature-based representation from structured attributes
* creating text embeddings from `canonical_text`
* calculating cosine similarity matrices
* combining feature-based and embedding-based similarities
* saving the final similarity matrix

### 2_inference.ipynb

This notebook contains the final recommendation logic.

Main functions:

* `recommend_hotels()` — returns top similar hotels for a selected hotel
* `compare_hotels()` — compares two hotels by attributes and service groups

The recommendation function uses a staged filtering logic. It first ranks all candidate hotels by similarity and then fills the final top recommendations through progressively relaxed business filters, such as geography, category, and known star rating.

## Methodology

### Feature-Based Similarity

Structured attributes are transformed into numerical vectors using:

* one-hot encoding for categorical features
* scaling for numerical features
* multi-label encoding for service lists

Cosine similarity is used to compare hotels based on these vectors.

### Embedding-Based Similarity

Each hotel is converted into a structured text representation called `canonical_text`.
Text embeddings are generated using a Sentence Transformers model, and cosine similarity is calculated between hotel embeddings.

### Combined Similarity

The final similarity matrix combines both approaches:

```text
final_similarity = 0.3 * feature_similarity + 0.7 * embedding_similarity
```

This allows the model to use both exact structured matches and semantic text similarity.

## Missing Data Handling

Missing information is not treated as a real negative signal.
For example, if a hotel has unknown star rating, this does not mean it belongs to the same group as another hotel with unknown star rating.

The project also calculates `comparison_data_share`, which shows how much information is available for comparing two hotels.

## Example Usage

```python
recommend_hotels(hotel_id=847, top_n=5)
```

```python
compare_hotels(hotel_id_1=847, hotel_id_2=900)
```

## Evaluation Approach

The dataset does not contain labeled “correct” recommendations, so the system is evaluated through qualitative and proxy checks:

* whether the source hotel is excluded from recommendations
* whether the required number of recommendations is returned
* whether there are no duplicate recommendations
* whether recommendations follow business logic
* how much the combined method differs from embedding-only recommendations
* manual comparison of recommended hotel pairs

## Limitations

* The project does not use user behavior or booking history.
* The recommendation system is content-based only.
* Some hotel attributes are incomplete or missing.
* There is no ground-truth relevance label for strict offline evaluation.
* Similarity weights were selected manually and can be further tuned.

## Technologies

* Python
* pandas
* NumPy
* scikit-learn
* Sentence Transformers
* Jupyter Notebook

## License

This project is licensed under the MIT License.

The original dataset is not included in the repository due to data access restrictions.
