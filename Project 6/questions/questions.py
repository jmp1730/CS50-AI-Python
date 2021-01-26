import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_contents = dict()

    for root, _, files in os.walk(directory):
        for file in files:
            f = open(os.path.join(root, file), "r")
            file_contents[file] = f.read()

    return file_contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.
    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    punctuation = string.punctuation
    stop_words = nltk.corpus.stopwords.words("english")

    words = nltk.word_tokenize(document.lower())
    words = [word for word in words if word not in punctuation and word not in stop_words]

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.
    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    total_num_documents = len(documents)
    words = set(word for sublist in documents.values() for word in sublist)
    
    for word in words:
        num_documents_containing_word = 0
        
        for document in documents.values():
            if word in document:
                num_documents_containing_word += 1
        
        idf = math.log(total_num_documents / num_documents_containing_word)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = dict()

    for file, words in files.items():
        total_tf_idf = 0
        for word in query:
            total_tf_idf += words.count(word) * idfs[word]
        file_scores[file] = total_tf_idf

    ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_files = [x[0] for x in ranked_files]

    return ranked_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = dict()

    for sentence, words in sentences.items():
        words_in_query = query.intersection(words)
        
        # idf value of sentence
        idf = 0
        for word in words_in_query:
            idf += idfs[word]
        
        # query term density of sentence
        num_words_in_query = sum(map(lambda x: x in words_in_query, words))
        query_term_density = num_words_in_query / len(words)

        # update sentence scores with idf and query term density values
        sentence_scores[sentence] = {'idf': idf, 'qtd': query_term_density}
    
    # rank sentences by idf then query term density
    ranked_sentences = sorted(sentence_scores.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)
    ranked_sentences = [x[0] for x in ranked_sentences]

    return ranked_sentences[:n]


if __name__ == "__main__":
    main()