Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
========================================================================================

* Authers: Yehuda Koren
* Proceedings of the 14th ACM SIGKDD (August 2008)

## Keywords

collaborative filtering, recommemder systems, explicit feedback, matrix factorization, neighborhood model, latent factor model, integrated model approach


## Summary

It is an integrated model approach, neighborbood approach and latent factor model(SVD), in CF field:

* Applying global optimization to neighborhood model by replacing user-specific interpolation weights into global weights, changing to sum over all items rated by user
* Considering implicit feeback in user latent factor vectors in latent factor model(SVD)
* Integrating two approach by sum of the prediction of ratings of them

The model parameters are determined by minimizing the associated regularized square error function through gradient descent

[Detailed Summary](https://www.notion.so/Factorization-Meets-the-Neighborhood-5014b81d066a4aca9fcae42aa40c9274)
