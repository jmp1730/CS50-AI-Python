import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # initial dict of applicable next pages given damping with initial value of 0
    states = corpus[page]

    # list of all possible pages from corpus
    all_pages = [i for i in corpus.keys()]

    if not not states:
        dict_state = dict.fromkeys(states, 0)

        for i in dict_state.keys():
            dict_state[i] = 1 / len(states) * damping_factor

        for i in all_pages:
            if i in dict_state.keys():
                dict_state[i] += 1 / len(all_pages) * (1 - damping_factor)
            else:
                dict_state[i] = 1 / len(all_pages) * (1 - damping_factor)

        return dict_state

    else:
        dict_state = dict.fromkeys(all_pages, 1/len(all_pages))
        return dict_state


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    all_pages = [i for i in corpus.keys()]

    # List of occurences
    occurences = []

    first = random.choice(all_pages)
    occurences.append(first)

    while len(occurences) < n:
        model_TM = transition_model(corpus, occurences[-1], damping_factor)
        next = np.random.choice(list(model_TM.keys()), 1, p=list(model_TM.values()))
        occurences.append(next[-1])

    sample_dict = dict.fromkeys(all_pages, 0)
    for i in sample_dict.keys():
        sample_dict[i] = occurences.count(i) / n

    return sample_dict



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = [i for i in corpus.keys()]
    iterate_dict = dict.fromkeys(all_pages, 1/len(all_pages))
    source_dict = dict.fromkeys(all_pages, {})

    for i in corpus.keys():
        if not corpus[i]:
            corpus[i] = set(all_pages)

    for i in all_pages:
        i_values = set()
        for j in corpus.keys():
            if i in corpus[j]:
                i_values.add(j)
        if not i_values or len(i_values) == 0:
            source_dict[i]  = set(all_pages)
        else:
            source_dict[i] = i_values

    while True:
        value = 0
        dict_copy = iterate_dict.copy()
        for key in dict_copy:
            iterate_dict[key] = PR(key, corpus, damping_factor, source_dict, dict_copy)
            if abs(dict_copy[key] - iterate_dict[key]) < 0.001:
                value += 1
        if value == len(all_pages):
            break

    return iterate_dict

def PR(page, corpus, damping_factor, source_dict, iterate_dict):
    damp_value = 0
    for parent in source_dict[page]:
        if len(corpus[parent]) == 0:
            x = len(corpus)
        else:
            x = len(corpus[parent])
        damp_value += iterate_dict[parent] / x

    new_value = (1 - damping_factor)/len(corpus) + damping_factor*damp_value
    return new_value




if __name__ == "__main__":
    main()
