import torch
import torchvision


def goodfellow(model, X, y):
	"""
	Use Goodfellow's trick to compute individual gradients.
	Ref: Efficient per-example gradient computations
	at: https://arxiv.org/abs/1510.01799
	"""
	model.zero_grad()

	logits, activations, linearCombs = model.forward(X)
	loss = F.binary_cross_entropy_with_logits(logits.view((-1,)), y)
	
	linearGrads = torch.autograd.grad(loss, linearCombs)
	gradients = goodfellow_backprop(activations, linearGrads)
	
	return gradients


def goodfellow_backprop(activations, linearGrads):
	grads = []
	for i in range(len(linearGrads)):
		G, X = linearGrads[i], activations[i]
		if len(G.shape) < 2:
			G = G.unsqueeze(1)
		
		G *= G.shape[0] # if the function is an average
		
		grads.append(torch.bmm(G.unsqueeze(2), X.unsqueeze(1)))
		grads.append(G)

	return grads
