% Context
% Amazon.com is one of the largest electronic commerce and cloud computing companies.
% Just a few Amazon related facts
%
% They lost 4.8 million in August 2013, when their website went down for 40 mins.
% They hold the patent on 1-Click buying, and licenses it to Apple.
% Their Phoenix fulfilment centre is a massive 1.2 million square feet.
%
% Amazon relies heavily on a Recommendation engine that reviews customer ratings and purchase history to recommend items and improve sales. 
% Content
% This is a dataset related to over 2 Million customer reviews and ratings of Beauty related products sold on their website.
% It contains 
%
% the unique UserId (Customer Identification), 
% the product ASIN (Amazon's unique product identification code for each product), 
% Ratings (ranging from 1-5 based on customer satisfaction) and 
% the Timestamp of the rating (in UNIX time)
%
% Acknowledgements
% A description of the entire Amazon products dataset.
% This dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014.
% This dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). 
% For the complete dataset check out Amazon Product Data
% Inspiration
% Can a good Recommendation engine be created from this minimal dataset?
% Give it a try!
% If you have got any more cool Amazon facts, dataset or queries, just drop a comment.
@RELATION Amazon---Ratings-(Beauty-Products)

@ATTRIBUTE UserId STRING
@ATTRIBUTE ProductId STRING
@ATTRIBUTE Rating REAL
@ATTRIBUTE Timestamp INTEGER

@DATA