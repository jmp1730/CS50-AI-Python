
import os
import random
import re
import sys
import math

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
    if corpus[page]:
        tot_prob = [(1 - damping_factor) / len(corpus)] * len(corpus)
        tot_prob_dict = dict(zip(corpus.keys(), tot_prob))
        link_prob = damping_factor / len(corpus[page])
        for link in corpus[page]:
            tot_prob_dict[link] += link_prob
        return tot_prob_dict
    else:
        return dict(zip(corpus.keys(), [1 / len(corpus)] * len(corpus)))


def sample_pagerank(corpus, damping_factor, n):
    pageranks = dict(zip(corpus.keys(), [0] * len(corpus)))
    page = random.choice(list(corpus.keys()))
    for _ in (range(n - 1)):
        pageranks[page] += 1
        prob_dist = transition_model(corpus, page, damping_factor)
        page = random.choices(list(prob_dist.keys()), prob_dist.values())[0]
    pageranks = {page: num_samp/n for page, num_samp in pageranks.items()}
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    tp = len(corpus)
    pageranks = dict(zip(corpus.keys(), [1/tp] * tp))
    pagerank1 = dict(zip(corpus.keys(), [math.inf] * tp))
    while any(pagerank1 > 0.001 for pagerank1 in pagerank1.values()):
        for page in pageranks.keys():
            link_prob = 0
            for link_page, links in corpus.items():
                if not links:
                    links = corpus.keys()
                if page in links:
                    link_prob += pageranks[link_page] / len(links)
            pagerank1 = ((1 - damping_factor) / tp) + (damping_factor * link_prob)
            pagerank1[page] = abs(pagerank1 - pageranks[page])
            pageranks[page] = pagerank1
    return pageranks

if __name__ == "__main__":
    main()