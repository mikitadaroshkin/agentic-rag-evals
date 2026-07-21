# Evaluation measures (information retrieval)

Evaluation measures for an information retrieval (IR) system assess how well an index, search engine, or database returns results from a collection of resources that satisfy a user's query. They are therefore fundamental to the success of information systems and digital platforms.
The most important factor in determining a system's effectiveness for users is the overall relevance of results retrieved in response to a query. The success of an IR system may be judged by a range of criteria including relevance, speed, user satisfaction, usability, efficiency and reliability. Evaluation measures may be categorised in various ways including offline or online, user-based or system-based and include methods such as observed user behaviour, test collections, precision and recall, and scores from prepared benchmark test sets.
Evaluation for an information retrieval system should also include a validation of the measures used, i.e. an assessment of how well they measure what they are intended to measure and how well the system fits its intended use case. Measures are generally used in two settings: online experimentation, which assesses users' interactions with the search system, and offline evaluation, which measures the effectiveness of an information retrieval system on a static offline collection.

Background
Indexing and classification methods to assist with information retrieval have a long history dating back to the earliest libraries and collections. Systematic evaluation of their effectiveness began in earnest in the 1950s with the rapid expansion in research production across military, government and education and the introduction of computerised catalogues. At this time, there were a number of different indexing, classification and cataloguing systems in operation which were expensive to produce and it was unclear which was the most effective.
Cyril Cleverdon, Librarian of the College of Aeronautics, Cranfield, England, began a series of experiments of print indexing and retrieval methods in what is known as the Cranfield paradigm, or Cranfield tests, which set the standard for IR evaluation measures for many years. Cleverdon developed a test called ‘known-item searching’ - to check whether an IR system returned the documents that were known to be relevant or correct for a given search. Cleverdon’s experiments established a number of key aspects required for IR evaluation: a test collection, a set of queries and a set of pre-determined relevant items which combined would determine precision and recall.
Cleverdon's approach formed a blueprint for the successful Text Retrieval Conference series that began in 1992.

Applications
Evaluation of IR systems is central to the success of any search engine including internet search, website search, databases and library catalogues. Evaluations measures are used in studies of information behaviour, usability testing, business costs and efficiency assessments. Measuring the effectiveness of IR systems has been the main focus of IR research, based on test collections combined with evaluation measures. A number of academic conferences have been established that focus specifically on evaluation measures including the Text Retrieval Conference (TREC), Conference and Labs of the Evaluation Forum (CLEF) and NTCIR.

Online measures
Online metrics are generally created from search logs. The metrics are often used to determine the success of an A/B test.

Session abandonment rate
Session abandonment rate is a ratio of search sessions which do not result in a click.

Click-through rate
Click-through rate (CTR) is the ratio of users who click on a specific link to the number of total users who view a page, email, or advertisement. It is commonly used to measure the success of an online advertising campaign for a particular website as well as the effectiveness of email campaigns.

Session success rate
Session success rate measures the ratio of user sessions that lead to a success. Defining "success" is often dependent on context, but for search a successful result is often measured using dwell time as a primary factor along with secondary user interaction, for instance, the user copying the result URL is considered a successful result, as is copy/pasting from the snippet.

Zero result rate
Zero result rate (ZRR) is the ratio of Search Engine Results Pages (SERPs) which returned with zero results. The metric either indicates a recall issue, or that the information being searched for is not in the index.

Offline metrics
Offline metrics are generally created from relevance judgment sessions where the judges score the quality of the search results. Both binary (relevant/non-relevant) and multi-level (e.g., relevance from 0 to 5) scales can be used to score each document returned in response to a query. In practice, queries may be ill-posed, and there may be different shades of relevance. For instance, there is ambiguity in the query "mars": the judge does not know if the user is searching for the planet Mars, the Mars chocolate bar, the singer Bruno Mars, or the Roman deity Mars.

Precision

Precision is the fraction of the documents retrieved that are relevant to the user's information need.

  
    
      
        
          
            precision
          
        
        =
        
          
            
              
                |
              
              {
              
                
                  relevant documents
                
              
              }
              ∩
              {
              
                
                  retrieved documents
                
              
              }
              
                |
              
            
            
              
                |
              
              {
              
                
                  retrieved documents
                
              
              }
              
                |
              
            
          
        
      
    
    {\displaystyle {\mbox{precision}}={\frac {|\{{\mbox{relevant documents}}\}\cap \{{\mbox{retrieved documents}}\}|}{|\{{\mbox{retrieved documents}}\}|}}}
  

In binary classification, precision is analogous to positive predictive value. Precision takes all retrieved documents into account. It can also be evaluated considering only the topmost results returned by the system using Precision@k.
Note that the meaning and usage of "precision" in the field of information retrieval differs from the definition of accuracy and precision within other branches of science and statistics.

Recall

Recall is the fraction of the documents that are relevant to the query that are successfully retrieved.

  
    
      
        
          
            recall
          
        
        =
        
          
            
              
                |
              
              {
              
                
                  relevant documents
                
              
              }
              ∩
              {
              
                
                  retrieved documents
                
              
              }
              
                |
              
            
            
              
                |
              
              {
              
                
                  relevant documents
                
              
              }
              
                |
              
            
          
        
      
    
    {\displaystyle {\mbox{recall}}={\frac {|\{{\mbox{relevant documents}}\}\cap \{{\mbox{retrieved documents}}\}|}{|\{{\mbox{relevant documents}}\}|}}}
  

In binary classification, recall is often called sensitivity. So it can be looked at as the probability that a relevant document is retrieved by the query.
It is trivial to achieve recall of 100% by returning all documents in response to any query. Therefore, recall alone is not enough but one needs to measure the number of non-relevant documents also, for example by computing the precision.

Fall-out
The proportion of non-relevant documents that are retrieved, out of all non-relevant documents available:

  
    
      
        
          
            fall-out
          
        
        =
        
          
            
              
                |
              
              {
              
                
                  non-relevant documents
                
              
              }
              ∩
              {
              
                
                  retrieved documents
                
              
              }
              
                |
              
            
            
              
                |
              
              {
              
                
                  non-relevant documents
                
              
              }
              
                |
              
            
          
        
      
    
    {\displaystyle {\mbox{fall-out}}={\frac {|\{{\mbox{non-relevant documents}}\}\cap \{{\mbox{retrieved documents}}\}|}{|\{{\mbox{non-relevant documents}}\}|}}}
  

In binary classification, fall-out is the opposite of specificity and is equal to 
  
    
      
        (
        1
        −
        
          
            specificity
          
        
        )
      
    
    {\displaystyle (1-{\mbox{specificity}})}
  
. It can be looked at as the probability that a non-relevant document is retrieved by the query.
It is trivial to achieve fall-out of 0% by returning zero documents in response to any query.

F-score / F-measure

The weighted harmonic mean of precision and recall, the traditional F-measure or balanced F-score is:

  
    
      
        F
        =
        
          
            
              2
              ⋅
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              ⋅
              
                r
                e
                c
                a
                l
                l
              
            
            
              (
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              +
              
                r
                e
                c
                a
                l
                l
              
              )
            
          
        
      
    
    {\displaystyle F={\frac {2\cdot \mathrm {precision} \cdot \mathrm {recall} }{(\mathrm {precision} +\mathrm {recall} )}}}
  

This is also known as the 
  
    
      
        
          F
          
            1
          
        
      
    
    {\displaystyle F_{1}}
  
 measure, because recall and precision are evenly weighted.
The general formula for non-negative real 
  
    
      
        β
      
    
    {\displaystyle \beta }
  
 is:

  
    
      
        
          F
          
            β
          
        
        =
        
          
            
              (
              1
              +
              
                β
                
                  2
                
              
              )
              ⋅
              (
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              ⋅
              
                r
                e
                c
                a
                l
                l
              
              )
            
            
              (
              
                β
                
                  2
                
              
              ⋅
              
                p
                r
                e
                c
                i
                s
                i
                o
                n
              
              +
              
                r
                e
                c
                a
                l
                l
              
              )
            
          
        
        
      
    
    {\displaystyle F_{\beta }={\frac {(1+\beta ^{2})\cdot (\mathrm {precision} \cdot \mathrm {recall} )}{(\beta ^{2}\cdot \mathrm {precision} +\mathrm {recall} )}}\,}
  

Two other commonly used F measures are the 
  
    
      
        
          F
          
            2
          
        
      
    
    {\displaystyle F_{2}}
  
 measure, which weights recall twice as much as precision, and the 
  
    
      
        
          F
          
            0.5
          
        
      
    
    {\displaystyle F_{0.5}}
  
 measure, which weights precision twice as much as recall.
The F-measure was derived by van Rijsbergen (1979) so that 
  
    
      
        
          F
          
            β
          
        
      
    
    {\displaystyle F_{\beta }}
  
 "measures the effectiveness of retrieval with respect to a user who attaches 
  
    
      
        β
      
    
    {\displaystyle \beta }
  
 times as much importance to recall as precision".  It is based on van Rijsbergen's effectiveness measure 
  
    
      
        E
        =
        1
        −
        
          
            1
            
              
                
                  α
                  P
                
              
              +
              
                
                  
                    1
                    −
                    α
                  
                  R
                
              
            
          
        
      
    
    {\displaystyle E=1-{\frac {1}{{\frac {\alpha }{P}}+{\frac {1-\alpha }{R}}}}}
  
.  Their relationship is:

  
    
      
        
          F
          
            β
          
        
        =
        1
        −
        E
      
    
    {\displaystyle F_{\beta }=1-E}
  
 where 
  
    
      
        α
        =
        
          
            1
            
              1
              +
              
                β
                
                  2
                
              
            
          
        
      
    
    {\displaystyle \alpha ={\frac {1}{1+\beta ^{2}}}}
  

Since F-measure combines information from both precision and recall it is a way to represent overall performance without presenting two numbers.

Average precision
Precision and recall are single-value metrics based on the whole list of documents returned by the system. For systems that return a ranked sequence of documents, it is desirable to also consider the order in which the returned documents are presented. By computing a precision and recall at every position in the ranked sequence of documents, one can plot a precision-recall curve, plotting precision 
  
    
      
        p
        (
        r
        )
      
    
    {\displaystyle p(r)}
  
 as a function of recall 
  
    
      
        r
      
    
    {\displaystyle r}
  
. Average precision computes the average value of 
  
    
      
        p
        (
        r
        )
      
    
    {\displaystyle p(r)}
  
 over the interval from 
  
    
      
        r
        =
        0
      
    
    {\displaystyle r=0}
  
 to 
  
    
      
        r
        =
        1
      
    
    {\displaystyle r=1}
  
:

  
    
      
        AveP
        =
        
          ∫
          
            0
          
          
            1
          
        
        p
        (
        r
        )
        d
        r
      
    
    {\displaystyle \operatorname {AveP} =\int _{0}^{1}p(r)dr}
  

That is the area under the precision-recall curve.
This integral is in practice replaced with a finite sum over every position in the ranked sequence of documents:

  
    
      
        AveP

---
Source: [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) — Wikipedia, licensed under CC BY-SA 4.0.
