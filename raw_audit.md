## Raw table audit

### `orders`

- Shape: 13084 rows × 9 columns
- Data types:

| Column | Data type | Missing count | Missing percent |
|---|---|---:|---:|
| `order_id` | `str` | 0 | 0.00% |
| `customer_id` | `str` | 0 | 0.00% |
| `product_id` | `str` | 0 | 0.00% |
| `quantity` | `int64` | 0 | 0.00% |
| `order_ts` | `str` | 0 | 0.00% |
| `status` | `str` | 0 | 0.00% |
| `discount_pct` | `int64` | 0 | 0.00% |
| `order_amount` | `str` | 53 | 0.41% |
| `delivery_minutes` | `float64` | 2541 | 19.42% |

### `customers`

- Shape: 1200 rows × 5 columns
- Data types:

| Column | Data type | Missing count | Missing percent |
|---|---|---:|---:|
| `customer_id` | `str` | 0 | 0.00% |
| `city` | `str` | 0 | 0.00% |
| `signup_date` | `str` | 0 | 0.00% |
| `age` | `float64` | 61 | 5.08% |
| `gender` | `str` | 30 | 2.50% |

### `products`

- Shape: 40 rows × 4 columns
- Data types:

| Column | Data type | Missing count | Missing percent |
|---|---|---:|---:|
| `product_id` | `str` | 0 | 0.00% |
| `product_name` | `str` | 0 | 0.00% |
| `category` | `str` | 0 | 0.00% |
| `price_inr` | `int64` | 0 | 0.00% |

### `tickets`

- Shape: 164 rows × 5 columns
- Data types:

| Column | Data type | Missing count | Missing percent |
|---|---|---:|---:|
| `ticket_id` | `str` | 0 | 0.00% |
| `customer_id` | `str` | 0 | 0.00% |
| `order_id` | `str` | 0 | 0.00% |
| `text` | `str` | 0 | 0.00% |
| `true_category` | `str` | 0 | 0.00% |

### Required raw-data figures

| Measure | Result |
|---|---:|
| Raw orders rows | 13,084 |
| Exact duplicate order rows | 119 |
| Rows with pending status | 656 |
| Distinct order_id values | 12,309 |
| Non-numeric order_amount entries | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

## Cleaning change counts

- Customers — city values standardized: **476**.
- Customers — gender values standardized: **950**.
- Customers — invalid ages converted to missing: **6**.
- Customers — ages imputed (original blanks plus invalid ages): **67**.
- Customers — signup dates converted: **1,200**.
- Orders — order timestamps converted: **13,084**.
- Orders — exact duplicate rows removed: **119**.
- Orders — older records removed after latest-per-order selection: **656**.
- Orders — unparseable amounts imputed from product price: **112**.
- Orders — negative delivery values set missing: **105**.
- Orders — completed delivery values imputed from product median: **774**.
- Orders — cancelled delivery values left missing: **1,199**.
