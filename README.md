# WebCrawling_Clustering

--README--
1. Open the project in VSCode or PyCharm
2. Run 'ThesisCrawler.py' found in the python folder
3. Enter when prompted:
   - Maximum files per year (e.g., 151)
   - Number of years to process (1-10)

You will be prompted to enter the maximum number of files to download per year and
the number of years to process starting from 2024.

***In the tests done, the ones with the results set aside in the current directory, 
the max files was set to 151 (which decreased to 120 due to invalid documents) 
and the number of years processed is 1. Not required to follow the same protocol***
------------------------------------------------------------------------------------------
The script generates in the 'python' folder:
- 'clustering_k{n}_results.txt': clustering analysis txt file
- 'clustering_silhouette_k{n}.png': Silhouette score visualizations png file
------------------------------------------------------------------------------------------
The results needed for instructions given in P2.pdf are in their respective folders.
The department/faculty clustering results are in 'k=4,k=38 Clustering Results'

The clustering results for 3 clusters and 6 clusters are in 'k=3,k=6 Clustering Results'
------------------------------------------------------------------------------------------

--Python Packages--
 1. Scrapy
- Installation: pip install scrapy
- Version: latest

 2. BeautifulSoup
- Installation: pip install beautifulsoup4
- Version: latest

 3. PyPDF2
- Installation: pip install PyPDF2
- Version: latest

 4. NLTK
- Installation: pip install nltk
- Version: latest
- Ensure nltk 'punkt' is downloaded

 5. Scikit-learn
- Installation: pip install scikit-learn
- Version: latest

6. Matplotlib
- Installation: pip install matplotlib
- Version: latest
- *.pyplot is important*

7. NumPy
- Installation: pip install numpy
- Version: latest
