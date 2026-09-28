-- your query
SELECT
    c.customer_id, c.name
FROM Customers c 
WHERE c.customer_id
NOT IN (
    SELECT
        c.customer_id
    FROM Customers c
    JOIN Orders o ON c.customer_id = o.customer_id
)
ORDER BY c.customer_id ASC