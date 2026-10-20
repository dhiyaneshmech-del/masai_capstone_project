-- Q1: completed customer totals for orders over INR 500
SELECT customer_id, SUM(order_amount) AS total_order_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_order_amount DESC;

-- Q2: completed revenue by city and product category
SELECT c.city, p.category, SUM(o.order_amount) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: products priced above average catalogue price
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: customers without any completed order
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: customers whose completed spend is above the average customer spend
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS average_total_spend
    FROM customer_spend
)
SELECT
    customer_spend.customer_id,
    customer_spend.total_spend,
    average_spend.average_total_spend
FROM customer_spend
CROSS JOIN average_spend
WHERE customer_spend.total_spend > average_spend.average_total_spend
ORDER BY customer_spend.total_spend DESC;

-- Q6: completed customer spending ranks within each city
WITH customer_spend AS (
    SELECT c.city, c.customer_id, SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT
    city,
    customer_id,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS city_row_number
FROM customer_spend
ORDER BY city, total_spend DESC, customer_id;

-- Q7: daily completed revenue with a cumulative total
WITH daily_revenue AS (
    SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
