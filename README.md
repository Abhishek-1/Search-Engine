# Vector-Space-Retrieval-System

The Cranfield collection is a standard IR text collection, consisting of 1400 documents from the
aerodynamics field, in SGML format. 

Tasks: For text pre-processing, remove stopwords, perform stemming (note: if a word becomes a stopword
after stemming, please remove it), remove punctuation and numbers (replace them with ""), split
on whitespace, and remove words with one or two characters in length. Perform the same text
processing operations on both the documents and the queries.

1. Implement an indexing scheme based on the vector space model. • TF-IDF
2. For each of the ten queries in the queries.txt le, determine a ranked list of documents, in descending order of their 
similarity with the query. The output of retrieval is a list of (query id, document id) pairs.

Determine of the average precision and recall for the ten queries, using:
• top 10 documents in the ranking
• top 50 documents in the ranking
• top 100 documents in the ranking
• top 500 documents in the ranking
