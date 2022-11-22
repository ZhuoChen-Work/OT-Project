# Optimal-Transport
This project is focused on finding an optimal transport mapping from Souce_distribution to Target_distribution.
We have built three models and compared their performance on different dataset (Gaussian, TwoMoon and MNIST).

1)ICNN-CTransform model: Based on the Brenier theorem, the optiaml transport plan should be the gradiant of an unique convex function.
ICNN(Input Convex Neural Network) is used to approximate this convex function.

2)LSE-CTransform model: Log Sum Exponential function can be used to approximate the convex function in some specific cases.

3)MLP-BarryCenter model: By computing the WasserStein Distance, we can know the ground truth optimal mapping between the training source and the training target. As one singe source point may be transported to multiple target points (s.t. the sum of the weights equal to one), the barycenter of these target points are choosed to be the reference points, which are used to compute the Mean Square Error.

# Part of Results (Toy example)
ICNN-CTransform on TwoMoon

souce: blue points;

target: orange points;

fake target: green points.


![moon](https://user-images.githubusercontent.com/118645613/203151908-fac52932-0f05-4d66-b5ec-6bbaf6e1d04d.gif)

ICNN-CTransform on MNIST

source: digits 8; 

target: digits 3;  

fake target: fake digits 3.

![MNIST](https://user-images.githubusercontent.com/118645613/203151876-cdd97694-0621-47d4-be29-43d258acd637.gif)
![MNIST](https://user-images.githubusercontent.com/118645613/203151906-4ed7c168-0ffb-4803-82c8-d4b3d0affe30.png)

