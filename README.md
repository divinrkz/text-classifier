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
For subnormal values between range $(0, 1)$, $s$ has to be positive, so $s = 0, e = 0, f \neq 0$ by definition of subnormal. Since the first bit of $f$ is $0$ in subnormal values, every value of $f$ is going to be less than $1$ therefore all values of $f$ can be represented as $2^{52}$ different numbers. Since the subnormal values can never have $1$ as value of $f$ by definition of subnormal numbers, then we only need to exclude the representation of $0$. Therefore, number of subnormal values that can be represented are $2^{52} - 1$.<br>
For normal values, $s = 0$, $e$ can be $2^{11} - 2$ different numbers, so for each $e$ we can have $2^{52}$ representation of $f$. By excluding 0, 1, by the rule of product, we can have a total of $(1022 \times 2^{52}) - 1$ numbers for normal values.
So, in total we have $(2^{52}-1) + ((1022 \times 2^{52}) - 1)$ numbers in range $(0,1)$
<br><br>
For subnormal values between range $(-\infty, 0)$, we have negative numbers so $s = 1$ and $e = 0$ by definition of subnormal numbers. So, we have $2^52$ different numbers for $f$. Since we have to exclude $0$, then the subnormal total numbers in range $(-\infty, 0) = 2^{52} - 1$. <br>
For normal values between range $(-\infty, 0)$, we have negative numbers only so $s = 1$, and since by definition of normal numbers $e \neq 0, e \neq 0x7ff$ so, it has $2^{11} - 2$ different numbers, and so since the first bit of normal numbers is $1$ we can't have $0$ value of $f$, so we can have $2^{52}$ different numbers on f. so the total subnormal total numbers in range $(-\infty, 0) = (2^{11} - 2) \times 2^{52}$<br>
Therefore the total values between range $(-\infty, 0) = (2^{52} - 1) + ((2^{11}-2) \times 2^{52})$

I claim that $ln(x)$ is an increasing function. <br>
Let $x_1, x_2 \in (0, \infty)$ such that $x_2 > x_1$.
Suppose $ln(x)$ is continuous over $(0, \infty)$ and differentiable over $(0, \infty).$<br>
Since $x_1, x_2 \in (0, \infty)$, then $ln(x)$ is also continuous over $[x_1, x_2]$ and differentiable over $[x_1, x_2].$<br>
By the Mean Value Theorem, there exists $c \in [x_1, x_2]$ such that $ln'(c) = \frac{ln(x_2) - ln(x_1)}{x_2 - x_1}$. <br>
Not that, $ln'(x) = \frac{1}{x}, \forall x \in (0, \infty)$. It follows that $\frac{1}{c} = \frac{ln(x_2) - ln(x_1)}{x_2 - x_1}$. <br>
Since $c \in [x_1, x_2]$ and $x_1, x_2 > 0$, therefore $c > 0$. <br>
It follows that, $\frac{1}{c} > 0 = \frac{ln(x_2) - ln(x_1)}{x_2 - x_1} > 0$. Since $x_2 > x_1$ by definition, $x_2 - x_1 > 0$ So, $ln(x_2) - ln(x_1) > 0 = ln(x_2) > ln(x_1)$<br>
Therefore, $ln(x)$ is an increasing function.
<br><br>
Note that, this number of probabilities we can represent increases when we take the $log$ because when you take the logarithm of probabilities where the domain between $0$ and $1$ (inclusive), you map them to a range where floating-point numbers are alot (densely packed). Since I've showed that the range $(-\infty, 0)$ has more floating point values than range $(0,1)$.
This transformation to be valid $log$ has to be monotonic. 
Observe that, suppose $p_1, p_2 \in [0,1]$ If probability $p_1 > p_2$ then $log(p_1) > log(p_2)$. Since comparisons are crucial for classification purposes, then $log$ should be monotonic because it maintains the order of probabilities after transformation.
