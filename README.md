char-rnn-implemented-in-python
---------------------------------
* RNN structure:
  * ht=tanh(Wxh*Xt+Whh *h(t-1)+bh)
  * yt=Why*ht+by

* Loss func:
  * Cross-entropy

* Tricks:
  * Cliping
  * Adagrad

* References:
  * [Stanford CS224D](http://cs224d.stanford.edu/syllabus.html)
  * [Adagrad](http://jmlr.csail.mit.edu/papers/v12/duchi11a.html)
  * [The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
