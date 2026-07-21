# Tf–idf

In information retrieval, tf–idf (term frequency–inverse document frequency, TF*IDF, TFIDF, TF–IDF, or Tf–idf) is a measure of importance of a word to a document in a collection or corpus, adjusted for the fact that some words appear more frequently in general. Like the bag-of-words model, it models a document as a multiset of words, without word order. It is a refinement over the simple bag-of-words model, by allowing the weight of words to depend on the rest of the corpus.
It was often used as a weighting factor in searches of information retrieval, text mining, and user modeling. A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries used tf–idf. Variations of the tf–idf weighting scheme were often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
One of the simplest ranking functions is computed by summing the tf–idf for each query term; many more sophisticated ranking functions are variants of this simple model.

Motivations
Karen Spärck Jones (1972) conceived a statistical interpretation of term-specificity called Inverse Document Frequency (idf), which became a cornerstone of term weighting:

The specificity of a term can be quantified as an inverse function of the number of documents in which it occurs.For example, the df (document frequency) and idf for some words in Shakespeare's 37 plays might be represented as follows:

We see that "Romeo", "Falstaff", and "salad" appears in very few plays, so seeing these words, one could get a good idea as to which play it might be. In contrast, "good" and "sweet" appears in every play and are completely uninformative as to which play it is.

Definition
The tf–idf is the product of two statistics, term frequency and inverse document frequency. There are various ways for determining the exact values of both statistics.
A formula that aims to define the importance of a keyword or phrase within a document or a web page.

Term frequency
Term frequency, tf(t,d), is the relative frequency of term t within document d, 

  
    
      
        
          t
          f
        
        (
        t
        ,
        d
        )
        =
        
          
            
              f
              
                t
                ,
                d
              
            
            
              
                ∑
                
                  
                    t
                    ′
                  
                  ∈
                  d
                
              
              
                
                  f
                  
                    
                      t
                      ′
                    
                    ,
                    d
                  
                
              
            
          
        
      
    
    {\displaystyle \mathrm {tf} (t,d)={\frac {f_{t,d}}{\sum _{t'\in d}{f_{t',d}}}}}
  
,
where ft,d is the raw count of a term in a document, i.e., the number of times that term t occurs in document d. Note the denominator is simply the total number of terms in document d (counting each occurrence of the same term separately). There are various other ways to define term frequency:

the raw count itself: tf(t,d) = ft,d
Boolean "frequencies": tf(t,d) = 1 if t occurs in d and 0 otherwise;
logarithmically scaled frequency: tf(t,d) = log (1 + ft,d);
augmented frequency, to prevent a bias towards longer documents, e.g. raw frequency divided by the raw frequency of the most frequently occurring term in the document:

  
    
      
        
          t
          f
        
        (
        t
        ,
        d
        )
        =
        0.5
        +
        0.5
        ⋅
        
          
            
              f
              
                t
                ,
                d
              
            
            
              max
              {
              
                f
                
                  
                    t
                    ′
                  
                  ,
                  d
                
              
              :
              
                t
                ′
              
              ∈
              d
              }
            
          
        
      
    
    {\displaystyle \mathrm {tf} (t,d)=0.5+0.5\cdot {\frac {f_{t,d}}{\max\{f_{t',d}:t'\in d\}}}}
  

Inverse document frequency

The inverse document frequency is a measure of how much information the word provides, i.e., how common or rare it is across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):

  
    
      
        
          i
          d
          f
        
        (
        t
        ,
        D
        )
        =
        log
        ⁡
        
          
            N
            
              n
              
                t
              
            
          
        
      
    
    {\displaystyle \mathrm {idf} (t,D)=\log {\frac {N}{n_{t}}}}
  

with

  
    
      
        D
      
    
    {\displaystyle D}
  
: is the set of all documents in the corpus

  
    
      
        N
        =
        
          
            |
          
          D
          
            |
          
        
      
    
    {\displaystyle N={|D|}}
  
: total number of documents in the corpus

  
    
      
        
          n
          
            t
          
        
        =
        
          |
        
        {
        d
        ∈
        D
        :
        t
        ∈
        d
        }
        
          |
        
      
    
    {\displaystyle n_{t}=|\{d\in D:t\in d\}|}
  
 : number of documents where the term 
  
    
      
        t
      
    
    {\displaystyle t}
  
 appears (i.e., 
  
    
      
        
          t
          f
        
        (
        t
        ,
        d
        )
        ≠
        0
      
    
    {\displaystyle \mathrm {tf} (t,d)\neq 0}
  
). If the term is not in the corpus, this will lead to a division-by-zero. It is therefore common to adjust the numerator to 
  
    
      
        1
        +
        N
      
    
    {\displaystyle 1+N}
  
 and the denominator to 
  
    
      
        1
        +
        
          n
          
            t
          
        
      
    
    {\displaystyle 1+n_{t}}
  
.

Term frequency–inverse document frequency

Then tf–idf is calculated as

  
    
      
        
          t
          f
          i
          d
          f
        
        (
        t
        ,
        d
        ,
        D
        )
        =
        
          t
          f
        
        (
        t
        ,
        d
        )
        ⋅
        
          i
          d
          f
        
        (
        t
        ,
        D
        )
      
    
    {\displaystyle \mathrm {tfidf} (t,d,D)=\mathrm {tf} (t,d)\cdot \mathrm {idf} (t,D)}
  

A high weight in tf–idf is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents; the weights hence tend to filter out common terms. Since the ratio inside the idf's log function is always greater than or equal to 1, the value of idf (and tf–idf) is greater than or equal to 0. As a term appears in more documents, the ratio inside the logarithm approaches 1, bringing the idf and tf–idf closer to 0.

Justification of idf
Idf was introduced as "term specificity" by Karen Spärck Jones in a 1972 paper. Although it has worked well as a heuristic, its theoretical foundations have been troublesome for at least three decades afterward, with many researchers trying to find information theoretic justifications for it.
Spärck Jones's own explanation did not propose much theory, aside from a connection to Zipf's law. Attempts have been made to put idf on a probabilistic footing, by estimating the probability that a given document d contains a term t as the relative document frequency,

  
    
      
        P
        (
        t
        
          |
        
        D
        )
        =
        
          
            
              
                |
              
              {
              d
              ∈
              D
              :
              t
              ∈
              d
              }
              
                |
              
            
            N
          
        
        ,
      
    
    {\displaystyle P(t|D)={\frac {|\{d\in D:t\in d\}|}{N}},}
  

so that we can define idf as

  
    
      
        
          
            
              
                
                  i
                  d
                  f
                
              
              
                
                =
                −
                log
                ⁡
                P
                (
                t
                
                  |
                
                D
                )
              
            
            
              
              
                
                =
                log
                ⁡
                
                  
                    1
                    
                      P
                      (
                      t
                      
                        |
                      
                      D
                      )
                    
                  
                
              
            
            
              
              
                
                =
                log
                ⁡
                
                  
                    N
                    
                      
                        |
                      
                      {
                      d
                      ∈
                      D
                      :
                      t
                      ∈
                      d
                      }
                      
                        |
                      
                    
                  
                
              
            
          
        
      
    
    {\displaystyle {\begin{aligned}\mathrm {idf} &=-\log P(t|D)\\&=\log {\frac {1}{P(t|D)}}\\&=\log {\frac {N}{|\{d\in D:t\in d\}|}}\end{aligned}}}
  

Namely, the inverse document frequency is the logarithm of "inverse" relative document frequency.
This probabilistic interpretation in turn takes the same form as that of self-information. However, applying such information-theoretic notions to problems in information retrieval leads to problems when trying to define the appropriate event spaces for the required probability distributions: not only documents need to be taken into account, but also queries and terms.

Link with information theory
Both term frequency and inverse document frequency can be formulated in terms of information theory; it helps to understand why their product has a meaning in terms of joint informational content of a document. A characteristic assumption about the distribution 
  
    
      
        p
        (
        d
        ,
        t
        )
      
    
    {\displaystyle p(d,t)}
  
 is that:

  
    
      
        p
        (
        d
        
          |
        
        t
        )
        =
        
          
            1
            
              
                |
              
              {
              d
              ∈
              D
              :
              t
              ∈
              d
              }
              
                |
              
            
          
        
      
    
    {\displaystyle p(d|t)={\frac {1}{|\{d\in D:t\in d\}|}}}
  

This assumption and its implications, according to Aizawa: "represent the heuristic that tf–idf employs."
The conditional entropy of a "randomly chosen" document in the corpus 
  
    
      
        D
      
    
    {\displaystyle D}
  
, conditional to the fact it contains a specific term 
  
    
      
        t
      
    
    {\displaystyle t}
  
 (and assuming that all documents have equal probability to be chosen) is:

  
    
      
        H
        (
        
          
            D
          
        
        
          |
        
        
          
            T
          
        
        =
        t
        )
        =
        −
        
          ∑
          
            d
          
        
        
          p
          
            d
            
              |
            
            t
          
        
        log
        ⁡
        
          p
          
            d
            
              |
            
            t
          
        
        =
        −
        log
        ⁡
        
          
            1
            
              
                |
              
              {
              d
              ∈
              D
              :
              t
              ∈
              d
              }
              
                |
              
            
          
        
        =
        log
        ⁡
        
          
            
              
                |
              
              {
              d
              ∈
              D
              :
              t
              ∈
              d
              }
              
                |
              
            
            
              
                |
              
              D
              
                |
              
            
          
        
        +
        log
        ⁡
        
          |
        
        D
        
          |
        
        =
        −
        
          i
          d
          f
        
        (
        t
        )
        +
        log
        ⁡
        
          |
        
        D
        
          |
        
      
    
    {\displaystyle H({\cal {D}}|{\cal {T}}=t)=-\sum _{d}p_{d|t}\log p_{d|t}=-\log {\frac {1}{|\{d\in D:t\in d\}|}}=\log {\frac {|\{d\in D:t\in d\}|}{|D|}}+\log |D|=-\mathrm {idf} (t)+\log |D|}
  

In terms of notation, 
  
    
      
        
          
            D
          
        
      
    
    {\displaystyle {\cal {D}}}
  
 and 
  
    
      
        
          
            T
          
        
      
    
    {\displaystyle {\cal {T}}}
  
 are "random variables" corresponding to respectively draw a document or a term. The mutual information can be expressed as

  
    
      
        M
        (
        
          
            T
          
        
        ;
        
          
            D
          
        
        )
        =
        H
        (
        
          
            D
          
        
        )
        −
        H
        (
        
          
            D
          
        
        
          |
        
        
          
            T
          
        
        )
        =
        
          ∑
          
            t
          
        
        
          p
          
            t
          
        
        ⋅
        (
        H
        (
        
          
            D
          
        
        )
        −
        H
        (
        
          
            D
          
        
        
          |
        
        W
        =
        t
        )
        )
        =
        
          ∑
          
            t
          
        
        
          p
          
            t
          
        
        ⋅
        
          i
          d
          f
        
        (
        t
        )
      
    
    {\displaystyle M({\cal {T}};{\cal {D}})=H({\cal {D}})-H({\cal {D}}|{\cal {T}})=\sum _{t}p_{t}\cdot (H({\cal {D}})-H({\cal {D}}|W=t))=\sum _{t}p_{t}\cdot \mathrm {idf} (t)}
  

The last step is to expand 
  
    
      
        
          p
          
            t
          
        
      
    
    {\displaystyle p_{t}}
  

---
Source: [Tf–idf](https://en.wikipedia.org/wiki/Tf–idf) — Wikipedia, licensed under CC BY-SA 4.0.
