import scrapy  # latest
from nltk import RegexpTokenizer
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup  # latest
import PyPDF2  # latest
from urllib.parse import urljoin
from collections import defaultdict
import os
from sklearn import metrics  # latest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt  # latest
from io import BytesIO
import numpy as np  # latest
import nltk  # latest

# Ensure the NLTK tokenizer is available
#nltk.download('punkt')


class ThesisCrawler(scrapy.Spider):
    name = "Thesis_Crawler"

    def __init__(self, max_files, num_years, *args, **kwargs):
        super(ThesisCrawler, self).__init__(*args, **kwargs)
        self.max_files = int(max_files)
        self.num_years = int(num_years)
        self.index = defaultdict(list)
        self.processed_urls = set()
        self.year_counters = defaultdict(int)  # Track documents per year
        self.global_counter = 0  # Track total documents across all years

    def start_requests(self):
        start_year = 2024
        for year in range(start_year, start_year - self.num_years, -1):
            # Construct URL for each year
            url = f'https://spectrum.library.concordia.ca/view/year/{year}.type.html'
            # Yield a request for each URL
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={'year': year})

    def parse(self, response, year):
        # Parse the HTML response with BeautifulSoup
        page_soup = BeautifulSoup(response.text, 'html.parser')

        # Find the thesis anchor found in the HTML body
        thesis_anchor = page_soup.find('a', attrs={'name': 'group_thesis'})
        if thesis_anchor:
            self.logger.info(f"Found 'group_thesis' anchor on {response.url}")

            # Iterate over all links following the thesis anchor
            for link in thesis_anchor.find_all_next('a', href=True):
                if self.year_counters[year] >= self.max_files:
                    break

                href = link.get('href')
                if href and href.startswith('http') and href not in self.processed_urls:
                    self.global_counter += 1
                    self.year_counters[year] += 1
                    doc_id = f"doc_{self.global_counter}"
                    self.logger.info(f"Following link for {doc_id} (Year {year}): {href}")
                    self.processed_urls.add(href)
                    yield scrapy.Request(href, callback=self.extract_pdf, cb_kwargs={'assigned_doc_id': doc_id})
        else:
            self.logger.info(f"No 'group_thesis' anchor found on {response.url}")

    def extract_pdf(self, response, assigned_doc_id):
        # Parse the HTML response text with BeautifulSoup
        page_soup = BeautifulSoup(response.text, 'html.parser')

        # Find citation block w
        citation_block = page_soup.find('span', class_='ep_document_citation')
        if citation_block:
            # Check for restricted access within the citation block
            restricted_text = citation_block.find(text=lambda t: t and "Restricted to Repository staff" in t)

            # Find the PDF links within the citation block
            pdf_link = citation_block.find('a', class_='ep_document_link')
            if pdf_link and pdf_link.get('href'):
                url = urljoin(response.url, pdf_link['href'])
                if url.endswith('.pdf') and url not in self.processed_urls:
                    self.processed_urls.add(url)

                    if restricted_text:
                        self.logger.info(f"Skipping restricted document {assigned_doc_id} on {response.url}")
                        return

                    self.logger.info(f"Found PDF link for {assigned_doc_id}: {url}")
                    yield scrapy.Request(url, callback=self.handle_pdf_response, cb_kwargs={'doc_id': assigned_doc_id})

    def handle_pdf_response(self, response, doc_id):
        # Get the content type of the response, figures out if pdf link is valid or not
        content_type = response.headers.get('Content-Type', b'').decode('utf-8', 'ignore').lower()

        try:
            if 'text/html' in content_type:
                if 'html' in response.text.lower():
                    self.logger.info(
                        f"Received HTML instead of PDF for {response.url}, likely a login page. Skipping {doc_id}")
                    return
        except AttributeError:
            pass

        if 'application/pdf' in content_type:
            self.download_and_parse_pdf(response, doc_id)  # Use original doc_id
        else:
            self.logger.info(f"Unexpected content type ({content_type}) for {response.url}. Skipping {doc_id}")

    def download_and_parse_pdf(self, response, doc_id):
        # Define the path to save the PDF
        pdf_path = f"{doc_id}.pdf"

        try:
            # Save the PDF to a file
            with open(pdf_path, 'wb') as f:
                f.write(response.body)

            # Extract text
            text = ""
            with open(pdf_path, 'rb') as f:
                try:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                        except Exception as e:
                            self.logger.warning(f"Error extracting text from page in {doc_id}: {str(e)}")
                            continue  # Skip problematic page and continue with next
                except Exception as e:
                    self.logger.error(f"Error reading PDF {doc_id}: {str(e)}")
                    return  # Skip this document entirely

            # Only process if we text is available
            if text:
                # Modified tokenizer to keep more terms, removing punctuation and numbers
                tokenizer = RegexpTokenizer(r'\b[a-zA-Z-]+\b')  # Allow hyphens within words

                # Tokenize and clean the text
                tokens = tokenizer.tokenize(text.lower())
                tokens = [token for token in tokens if not token.isdigit()]

                # Update the inverted index
                for token in tokens:
                    if doc_id not in self.index[token]:
                        self.index[token].append(doc_id)
            else:
                self.logger.warning(f"No text extracted from {doc_id}")

        except Exception as e:
            self.logger.error(f"Error processing {doc_id}: {str(e)}")
        finally:
            # Delete pdf off of directory
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as e:
                self.logger.error(f"Error removing temporary file {pdf_path}: {str(e)}")

    def closed(self, reason):
        if self.index:
            # Uncomment the following to print the index to the console
            #print("\nInverted Index:")
            #for token, doc_ids in sorted(self.index.items()):
            #   print(f"Token: {token}, Document IDs: {doc_ids}")

            # Get unique document IDs from the index
            all_doc_ids = set()
            for doc_ids in self.index.values():
                all_doc_ids.update(doc_ids)

            # Prepare documents for clustering
            documents = []
            doc_ids = []
            for doc_id in sorted(all_doc_ids):
                # Get all tokens for this document
                doc_tokens = [token for token, docs in self.index.items()
                              if doc_id in docs]

                if doc_tokens:
                    documents.append(" ".join(doc_tokens))
                    doc_ids.append(doc_id)

            print(f"\nNumber of documents to cluster: {len(documents)}")


            def perform_clustering(k, cluster_type):
                # Ensure k is not larger than the number of documents
                actual_k = min(k, len(documents)-1)
                if actual_k != k:
                    print(f"\nNote: Adjusted number of clusters from {k} to {actual_k} due to document sample size")

                print(f"\n{'=' * 80}")
                print(f"Clustering Analysis with k={actual_k} ({cluster_type})")
                print(f"{'=' * 80}")

                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(
                    max_df=1.0,
                    min_df=1,
                    stop_words='english',
                    token_pattern=r'\b[a-zA-Z-]+\b',  # Allow letters and hyphens
                    lowercase=True  # Keep lowercased
                )
                X_tfidf = vectorizer.fit_transform(documents)
                print(f"TF-IDF matrix shape: {X_tfidf.shape}")

                # LSA transformation
                n_components = min(actual_k, X_tfidf.shape[1] - 1)
                lsa = make_pipeline(
                    TruncatedSVD(n_components=n_components),
                    Normalizer(copy=False)
                )
                X_lsa = lsa.fit_transform(X_tfidf)

                # K-means clustering
                kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=5)
                kmeans.fit(X_lsa)

                # Analyze clusters
                clusters = {}
                for doc_id, cluster_id in zip(doc_ids, kmeans.labels_):
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(doc_id)

                # Get feature names (terms)
                terms = vectorizer.get_feature_names_out()

                # Calculate overall silhouette score
                silhouette_avg = metrics.silhouette_score(X_lsa, kmeans.labels_)
                print(f"\nOverall Silhouette Score: {silhouette_avg:.3f}")

                # Calculate silhouette scores for individual samples
                sample_silhouette_values = metrics.silhouette_samples(X_lsa, kmeans.labels_)

                # Calculate and store cluster-specific metrics
                cluster_metrics = {}

                for cluster_id in range(actual_k):
                    # Get indices for this cluster
                    cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]

                    # Calculate cluster-specific silhouette score
                    clustering_silhouette = np.mean(sample_silhouette_values[cluster_indices])

                    # Store metrics
                    cluster_metrics[cluster_id] = {
                        'size': len(cluster_indices),
                        'silhouette': clustering_silhouette
                    }

                    # To print cluster metrics in console
                    #print(f"\nCluster {cluster_id}:")
                    #print(f"Size: {cluster_metrics[cluster_id]['size']} documents")
                    #print(f"Average Silhouette Score: {clustering_silhouette:.3f}")

                # Create output file for this clustering run
                with open(f'clustering_k{actual_k}_results.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Clustering Analysis with k={actual_k} ({cluster_type})\n")
                    f.write(f"{'=' * 80}\n")
                    f.write(f"Number of documents: {len(documents)}\n")
                    f.write(f"TF-IDF matrix shape: {X_tfidf.shape}\n")
                    f.write(f"Overall Silhouette Score: {silhouette_avg:.3f}\n\n")

                    # Write cluster metrics
                    f.write("\nCluster Metrics:\n")
                    f.write("-" * 40 + "\n")
                    for cluster_id, metrics_dict in cluster_metrics.items():
                        f.write(f"\nCluster {cluster_id}:\n")
                        f.write(f"Size: {metrics_dict['size']} documents\n")
                        f.write(f"Average Silhouette Score: {metrics_dict['silhouette']:.3f}\n")

                    # For each cluster
                    for cluster_id in range(actual_k):
                        f.write(f"\nCluster {cluster_id} Detailed Analysis:\n")
                        f.write("-" * 40 + "\n")

                        # Documents in this cluster
                        cluster_docs = clusters.get(cluster_id, [])
                        f.write(f"Documents: {cluster_docs}\n")

                        # Calculate TF-IDF scores for terms in this cluster
                        term_scores = defaultdict(float)
                        term_count = defaultdict(int)

                        for doc_id in cluster_docs:
                            doc_idx = doc_ids.index(doc_id)
                            doc_vector = X_tfidf[doc_idx]

                            for idx, score in zip(doc_vector.indices, doc_vector.data):
                                term = terms[idx]
                                term_scores[term] += score
                                term_count[term] += 1

                        # Calculate average scores and sort
                        cluster_terms = [
                            (term, score / term_count[term])
                            for term, score in term_scores.items()
                        ]
                        cluster_terms.sort(key=lambda x: x[1], reverse=True)

                        # Get top XX terms, change the value of 50 to
                        f.write("\nTop 50 Most Informative Terms:\n")
                        f.write("Rank\tTerm\t\tTF-IDF Score\tTF-IDF Rank\n")
                        f.write("-" * 60 + "\n")

                        # Create a mapping of terms to their TF-IDF ranks
                        tfidf_ranks = {term: rank for rank, (term, _) in enumerate(cluster_terms, 1)}

                        # Output top 50 terms with their TF-IDF ranks
                        for rank, (term, score) in enumerate(cluster_terms[:50], 1):
                            tfidf_rank = tfidf_ranks[term]
                            f.write(f"{rank}\t{term:<15}\t{score:.4f}\t\t{tfidf_rank}\n")

                # Visualize cluster silhouette scores
                plt.figure(figsize=(10, 6))
                clustering_silhouette_scores = [metrics['silhouette'] for metrics in cluster_metrics.values()]
                plt.bar(range(actual_k), clustering_silhouette_scores, color='skyblue')
                plt.axhline(y=silhouette_avg, color='red', linestyle='--', label='Overall Average')
                plt.xlabel('Cluster ID')
                plt.ylabel('Silhouette Score')
                plt.title(f'Silhouette Scores per Cluster (k={actual_k}, {cluster_type})')
                plt.legend()
                plt.savefig(f'clustering_silhouette_k{actual_k}.png')
                plt.close()


            n_docs = len(documents)
            print(f"\nTotal documents available for clustering: {n_docs}")

            if n_docs >= 2:
                # Change the value of k to be however many clusters needed
                # If num of clusters >= num of documents then, num of clusters = num of documents - 1
                perform_clustering(38, "Departments")
                perform_clustering(4, "Faculties")
                # It's possible to add or remove clustering as desired
            else:
                print("Not enough documents for meaningful clustering (minimum 2 required)")
        else:
            print("Inverted Index is empty.")


def run_spider(max_files, num_years):
    process = CrawlerProcess(settings={
        "USER_AGENT": "Chrome/131.0.6778.86 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "DOWNLOAD_DELAY": 1,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "ROBOTSTXT_OBEY": True,
        "LOG_LEVEL": "INFO",
        "HTTPERROR_ALLOW_ALL": True,
        "DOWNLOAD_FAIL_ON_DATALOSS": False,
        "DOWNLOAD_WARNSIZE": 0
    })
    process.crawl(ThesisCrawler, max_files=max_files, num_years=num_years)
    process.start()


if __name__ == "__main__":
    while True:
        try:
            max_files = int(input("Enter the maximum number of files to download per year: "))
            if max_files > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    while True:
        try:
            num_years = int(input("Enter the number of years to process starting from 2024: "))
            if 1 <= num_years <= 10:  # Limiting to 10 years maximum
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Please enter a valid number.")

    run_spider(max_files, num_years)
    print("Process Finished. View directory for clustering & silhouette evaluation")