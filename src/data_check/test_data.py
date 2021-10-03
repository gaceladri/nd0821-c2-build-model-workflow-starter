"""
Some test about "data quality"
Author: Adrian
Date: 3 Oct 2021
"""

import pandas as pd
import scipy


def test_column_names(data):
    """
    Test that we have the expected column names
    """

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):
    """
    Test unique names in neighborhood column
    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC
    """
    assert len(data[data.longitude < -74.3]) == 0
    assert len(data[data.longitude > -73.5]) == 0
    assert len(data[data.latitude > 40.95]) == 0
    assert len(data[data.latitude < 40.4]) == 0


def test_similar_neigh_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data: pd.DataFrame):
    """
    Test that we don't have too much values or too few
    """
    assert 1500 < data.shape[0] < 1000000


def test_price_range(data: pd.DataFrame):
    """
    Test that we have filtered the prices by the upper interquantile 99% -> 1800
    """
    assert data.price.max() == 1800
    assert data.price.min() == 0
