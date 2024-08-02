### <span style="color:DarkSlateBlue">Information about the Modeling Process</span>

Recommender systems can broadly be classified as collaboration-based, content-based, or a hybrid of the two. Collaboration-based recommender systems find similarities between users and use these similarities to recommend items that similar users would like. On the other hand, content-based recommender systems find similarities between items to recommend items similar to items users like. A hybrid system combines the previous two systems.

For **Recommend Anime**, we implemented a collaborative-based recommender system, leveraging user rating similarities to suggest anime. Given the large size of the anime and user rating datasets, we needed an algorithm capable of efficiently processing substantial amounts of data to uncover patterns between users and predict ratings.

We opted for a matrix factorization algorithm, specifically Singular Value Decomposition (SVD). Matrix factorization uses linear algebra techniques to reduce data dimensionality, making it easier to identify hidden patterns. SVD was selected over other methods like SVD++ and Non-Negative Matrix Factorization (NMF) due to its lower error rate and faster training and prediction times. This choice was based on a comparative analysis of these algorithms, ensuring that our recommendations are both accurate and efficient.

### <span style="color:DarkSlateBlue">Future Work</span>
In the future, we plan to enhance **Recommend Anime** with the following features:

* User registration, login, and authentication.
* Enhanced recommendation customization options.
* Integration into the **Anime-Flix** streaming app for seamless viewing and recommendations.

---

<div style='text-align: center;'>
    <p>Developed by Kawaii Consultants | Contact us at: <a href="mailto:info@kawaiiconsultants.com">info@kawaiiconsultants.com</a></p>
</div>