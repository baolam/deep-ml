-- your query
SELECT
    day,
    SUM (CASE WHEN id % 2 <> 0 THEN measurement else 0 END) as odd_sum,
    SUM (CASE WHEN id % 2 == 0 THEN measurement else 0 END) as even_sum
FROM readings
GROUP BY day
ORDER BY day ASC