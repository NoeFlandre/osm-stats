TOP_KEYS_QUERY = """
SELECT 
    key, 
    count_all 
FROM keys 
ORDER BY count_all DESC 
LIMIT 10;
"""

TOP_TAGS_QUERY = """
SELECT 
    key, 
    value, 
    count_all 
FROM tags 
ORDER BY count_all DESC 
LIMIT 10;
"""

METADATA_QUERY = """
SELECT * FROM source;
"""

GLOBAL_KEY_AGGREGATES_QUERY = """
SELECT 
    COUNT(key) AS total_distinct_keys, 
    SUM(count_all) AS total_key_occurrences 
FROM keys;
"""

GLOBAL_TAG_AGGREGATES_QUERY = """
SELECT 
    COUNT(*) AS total_distinct_tags, 
    SUM(count_all) AS total_tag_occurrences 
FROM tags;
"""
