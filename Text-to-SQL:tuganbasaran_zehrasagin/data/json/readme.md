# Data Processing Challenge
Thank you for your interest in the opportunity at Safeguard Global.

As we seek to understand your skills better, we request your participation in a take-home task. This challenge is designed to evaluate your proficiency in sourcing, transforming, and querying data.

## Instructions & Submission
- Please submit your solution as a Python script named `sg_data_challenge.py`.
- Define separate functions for each task.
- Include unit tests if applicable.
- Include a short readme for your solution named `sg_data_challenge.md`.
- Your submission should include a `main()` function that runs the unit tests (if any) followed by executing all the tasks in sequence.

## Notice
This document and the data provided for this challenge are **proprietary**. Please do not share them with anyone or post the questions or your answers anywhere online, including on any Git repositories.

## Tasks
Given two JSON files, one containing data for "products" and the other for "orders," your objective is to create functions to solve the following tasks:

1. Calculate the net sales amount. Note that there are three order statuses: created, cancelled, and returned.

1. Calculate the average net sales price.

1. Calculate the gross (total) sales amount.

1. Calculate the average gross (total) sales price.

1. Determine the average sales amount for the last 5 days of sales. Note that the sales do not have to be over 5 consecutive calendar days.

1. Identify the location with the highest sales.

1. In the `orders` dataset, for each `product_id`, calculate the price change (i.e., if the price of the order is increased, you can write `rise`. If the price of the order is decreased, you can write `fall`).

    Sample Input:
    ```
    +--------+------+-------------------+
    |prod_id |price |order_date         |
    +--------+------+-------------------+
    |<id-1>  |3.00  |2021-01-22 01:20:32|
    |<id-1>  |3.00  |2021-01-22 02:50:20|
    |<id-1>  |3.25  |2021-01-22 03:45:10|
    |<id-2>  |3.25  |2021-01-22 13:45:10|
    |<id-2>  |3.25  |2021-01-22 14:45:10|
    |<id-2>  |3.45  |2021-01-22 15:45:10|
    |<id-1>  |3.25  |2021-01-22 04:57:24|
    |<id-1>  |2.99  |2021-01-22 05:44:47|
    |<id-1>  |2.99  |2021-01-22 06:34:43|
    |<id-1>  |3.50  |2021-01-22 07:05:29|
    +--------+------+-------------------+
    ```

    Sample Output:
    ```
    +--------+------+-------------------+--------+
    |prod_id |price |order_date         |change  |
    +--------+------+-------------------+--------+
    |<id-1>  |3.25  |2021-01-22 03:45:10|rise    |
    |<id-1>  |2.99  |2021-01-22 05:44:47|fall    |
    |<id-1>  |3.50  |2021-01-22 07:05:29|rise    |
    |<id-2>  |3.45  |2021-01-22 15:45:10|fall    |
    +--------+------+-------------------+--------+
    ```

1. Which products were ordered in the same year as their release date?
1. Visualize the average price per release year for each location using the most suitable chart.
1. Visualize the distribution of weekly gross (total) sales amount. Does the distribution resemble a normal distribution?
1. Visualize gross (total) sales amount per week and highlight anomalies on the chart.
