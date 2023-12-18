# Text Classification. 

A common machine learning problem we encounter is wanting to classify documents of words into different
categories. It would be great, for example, to be able to classify an email as “spam” without reading. 


Naive Bayes is what is called a “bag of words” approach, which reduces a document to just a set of words
(meaning that we are completely ignoring the order of the words) and works with that. This seems like it loses
the majority of the information—and it does—but it turns out to work decently well despite this.
Let’s say we have a document with n distinct words, which we enumerate arbitrarily as $W = {w_i}^{n−1}_{i=0}$ (note that
this is not necessarily the order of the words in the document). We can imagine that the document (defined for
our purposes by the set $W$) was pulled from some ideal distribution of documents, and let $B_{word}$ be the event
that a document pulled from this distribution contains word. Let $A$ be the event that our document is spam.
In order to classify our document as spam or not spam, we want to find $Pr(A | B)$, where
$$ B = \bigcap^{n-1}_{i=0}B_{w_{i}} $$ 
is the event that a document drawn from the distribution of documents contains all of the words $W$.
Applying Bayes’ Theorem gives us 
$$ Pr(A|B) = \frac{Pr(B|A)Pr(A)}{Pr(B)}$$ 

----------

### Claim
A proportional likelihood value suffices for classification purposes, and we
don’t need to calculate the actual value of $Pr(A | B)$, So,
$$Pr(A | B) \propto Pr(B | A) Pr(A)$$

### Proof
Note that, when classifying, we are interested in comparing the likelihoods of  categories $C_k$ to determine which is most probable that is to determine which has the largest probability, not the actual values.\\
Since the denominator of Naive Bayes' Theorem $Pr(B)$ is constant for all categories, it does not affect the comparison and thus can be ignored for classification. We use the numerator of the Bayes' theorem, to classify the document because it varies with the $C_k$.

Suppose, $$Pr(A_1) = \frac{P(B|A_1)Pr(A_1)}{Pr(B)} > Pr(A_2) = \frac{P(B|A_2)Pr(A_2)}{Pr(B)}$$
Note that, if the numerator for $Pr(A_1) > $ $Pr(A_2)$, then the inequality will always hold. 
So, $$Pr(A_1) = {P(B|A_1)Pr(A_1)} > Pr(A_2) = {P(B|A_2)Pr(A_2)}.$$ It follows that regardless of the values of $Pr(B)$, the comparison is not affected that is why only the proportional likelihood is necessary for classification.

----------
It turns out, real numbers are not stored exactly on computers. Instead, computers use a system called “floating
point numbers” which is a standardized approximation of the real numbers and used everywhere. At a high
level, floating point numbers are basically a binary version of scientific notation.
To fix this underflow problem, we’re going to be taking the log of the probabilities and using those instead, like
so: 

### Claim
$
\begin{equation}
    \begin{split}
Pr(A | B) &\propto \left(\prod^{n}_{i=0}Pr\left(B_{w_{i}} | A\right)\right)
 \\
log\left(Pr(A | B)\right) &= log\left(\left(\prod^{n}_{i=0}Pr\left(B_{w_{i}} | A\right)\right)Pr(A) \right) \\
&= \left( \sum^{n}_{i=0} log(Pr\left(B_{w_{i}} | A\right))\right)+ log \left(Pr(A) \right)
    \end{split}
\end{equation}
$

### Proof
