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
