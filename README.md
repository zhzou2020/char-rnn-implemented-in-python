char-rnn-implemented-in-python
---------------------------------
* RNN structure:<br/>
  * ht=tanh(Wxh*Xt+Whh *h(t-1)+bh)<br/>
  * yt=Why*ht+by<br/>

* Loss func:<br/>
  * cross-entropy<br/>

* Tricks:
  * cliping
  * Adagrad

* References:
  * [stanford CS224D](http://cs224d.stanford.edu/syllabus.html)
  * [Adagrad](http://jmlr.csail.mit.edu/papers/v12/duchi11a.html)
  * [The Unreasonable Effectiveness of Recurrent Neural Networks by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
