# Triumph-Sportswear-Logistic-Regression-Based-Classification

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/af550b72-ccf0-4eed-8dfb-88bcdde99e15)


**=>Introduction**

Triumph as a company has two business model wherein they sell their products directly to
other business and one where they sell to consumers. The B2C happens in two mediums one
through Amazon and one through their own website. One of the major issue they face is huge
number of products sold on amazon are returned and due to Amazon’s no question asked
policy they are not able to know the reason for which it has been returned. Hence my project 
tried to help them by building a classification model based on Machine
learning which will help them to predict accurately if the product is returned or will it be
shipped successfully.

In the following project I tried to approach the problem as a basic classification problem by
selecting my target variable as the Transaction Type column in the dataset. The dataset is mix
of sales happening and detailed description of every item. I was given with the taxes
and where the item is selling and shipping to. I was given with the state to which the
item is shipping and the type of the transaction. Further I analysed the data and have
found some of the valuable insights which can help the company and its marketing team to
better design plan for the promotion strategies and also will help them to consider their
inventory while preparing for the orders.

**=> Data Overview**

I asked Triumph, a Sportswear Company for their data and they provided me with their data
related to Amazon sales from August 2021 to March 2022. The company basically
manufactures sportswear and they even sell their products business-to-consumer (B2C)
through Amazon as a mediator. The company has effectively managed and made proper
entries with valid details related to their Amazon sales. Their data includes

• Seller Gstin – GST number of the seller

• Invoice Number – Unique number assigned to invoices

• Invoice Date- Date and Time when the product was billed

• Transaction Type- Return, Shipment, FreeReplacement or Cancel

• Order Id, Shipment Id and Shipment Item Id- Unique Ids of Order, Shipment and
Shipment Item respectively

• Shipment Date- Date and Time of the Product Shipped

• Order Date- Date and Time of the Order

• Quantity- Quantity of the item ordered

• Item Description-Description of the item

• Product Tax Code- Tax Code of the Product

• Columns carrying details of where the product was billed such as Bill from City,
State, Country and Postal Code.

• Columns carrying details from where the product was shipped such as Ship from City,
State, Country, Postal Code and even carried details to where the product was shipped such
as Ship to City, State, Country and Postal Code.

• Invoice Amount- The final amount the buyer has to pay

• Tax Exclusive Gross- Gross amount without tax

• Total Tax Amount-Total amount of taxes

• Columns carrying details of taxes rate such as Cgst rate, Sgst rate, Utgst rate, Igst rate
etc.

• Columns carrying data of Gift Wrap, Shipping Promo Discount basis etc.

• Payment Method Code- Method by which the payment was done

• Credit Note No – Number of Credit Note
• Credit Note Date- Date and time of Credit Note

**=>Data cleaning**

The practice of correcting or deleting incorrect, corrupted, improperly formatted, duplicate,
or incomplete data from a dataset is known as data cleaning. There are numerous ways for
data to be duplicated or mislabelled when merging multiple data sources. Even if the data is
right, outcomes and algorithms are untrustworthy if the data is erroneous. Because the
methods will differ from dataset to dataset, there is no one-size-fits-all approach to
prescribing the exact phases in the data cleaning process. Different steps involved in cleaning
data are to remove duplicate or irrelevant observations, fix structural error, filter unwanted
outliers and handle missing data.

There were some rows in our data which were empty or NaN. So to solve the problem I
imputed the mean of the respective columns in that row. But I was facing one problem in
that approach. Due to outliers present in data this approach was affecting our analysis. So I
decided to impute median instead of mean in our approach. I removed the ID column from
our dataset as it was not important in the analysis part. Then we club(merge) different
columns of taxes in our dataset to one for the ease of analysis. Different taxes in our dataset
are CGST, IGST, SGST and UTGST. I also removed the rows where the order was
cancelled as it was not affecting our analysis.

Then I used data encoding. As I was using Logistic regression in our approach it was
necessary to convert text or categorical data into numbers. In data science and Machine
learning this is known as Data encoding. There are mainly two types of encoding: Label and
one hot encoding. In one hot encoding each category is represented as one hot vector.
Suppose there are three countries in one column in a dataset which are India, US and Japan.
Then we will make three columns of India, US and Japan and place 0 if that row does not
belong to a specific country and place 1 if the row belongs to that specific country. In Label
encoding each row is assigned an integer based on category. For example, the US as 0 , India
as 1 and Japan as 2.

**=> Calculations**

**TFIDF Vectorisation Calculation and Explanation:**

This is used in Natural language processing to assign the number to categorical features.
Difference between TFIDF and label encoding is that in TFIDF the words are assigned
weights. We will see in further explanation. Term Frequency is the occurrences of a word(w)
in document d by total number of words in document d.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/99474a80-f29a-4e52-8ff5-21ece2292991)

Let’s take an example. Suppose in a corpus D there are two documents A and B.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/20ca1b59-af12-4eb4-ab7c-08dc0c0fe37c)

From the table we can see that TF is more for the words whose frequency in the document is
more. But this does not give the importance of that word. For that we need Inverse Document
Frequency (IDF).

Some words like (of, the) are more frequently used but are of less importance. So, the job of
the IDF is to provide weightage to every word based on the frequency.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/20818408-32df-4569-b1fd-2e330bcf4098)

**=>My approach of TFIDF Model:**

In my dataset I was having Item description columns. It was very difficult to encode
categorical data into numerical. So I used TF IDF from sklearn library of python to tackle
this problem. As explained above the algorithm basically converted the text from the column
and made different 659 columns of words for e.g. ‘active’, ’activewear’, ‘adul’, ‘adult’ and
assigned weights as explained above. This helped to achieve good accuracy for my model
and handle many categorical variables without dropping the Item Description column. Below
is the picture of my code.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/a09a1250-025a-4802-81d2-44576b775ed6)

**=>Observations**

From the data, I have seen that the sales of the product were more during December and
January. The reason behind the increase in the sales of the product from December to January
was that the competition held during this time was more. As the competition are more, the
apparels required for it will be more compared to other time during the year. So, there will be
increase in the sales of the product from December to January. The other thing which we
have observed from the data was the Cycling apparels were the most selling items. The
cycling jersey, shots and the helmet are most selling items in this company. Also, the other
thing which we have observed from the data was that the company was selling skates. This
was the only sports instrument which the company was selling. So, due to this there was more
sales of this product.

Also, there was a trend seen in the data between festivals and refund. Sometimes, people buy
the product for the festival and after using it they will refund the product. People order the
jersey of Indian flag one or two day before the Independence Day and after wearing it on the
Independence Day, they will refund the jersey. The reason behind this is Amazon is not
asking many questions for refunding the product. So, one will do that they don’t like the
material, the size was not perfect and many more reasons for refunding the product. This type
of thing also happens during the time of Mahashivratri. Another thing which I observed
from the data was that the customer cancels their product when they are using the payment
mode of cash on Delivery. So, in order to resolve this type of problems, Amazon must have
proper policy for refunding the product.

The main thing of this company was that the company was falling under the facility of
Amazon prime services. So, there were certain rules and regulations which must be followed.
The rule was that if customer places the order today than they should get their item by
tomorrow. But here the thing is that if the customer orders the product till 13:15 am, then that
item should be compulsory shipped at 13:30 am. After 13:16 am, the order received to the
company is to shipped the next day. If the company is not able to shipped the item ordered till
13:15 am, then the company will be removed from the facility of Amazon prime services.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/131382b5-6780-4971-97a8-60c98edad786)

**=>Findings from the data**

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/201c699a-7378-47ac-9b6d-cf2374163e79)

The following chart represents the sales happening on an hourly chart. The following data
was extracted from the order-date by using suitable functions. From this graph, we can see
that sales are high at 7 o’clock in the morning. The reason behind this can be the company
selling sports apparel. As the maximum sales happen at 7 o’clock in the early morning, the
buyers may be health-conscious. The buyers might go for exercise or might be he/she playing
any kind of sports so he/she may order sports apparel like cycling shorts, t-shirts, etc. From
9:00 am to 1:00 the sales remain almost the same. After 1 o’clock for a few hours one can see
that the sales decrease, the reason behind this can be the Prime membership. Due to same-day
delivery service for Prime users, the order has to be placed before 1:15 pm, so the sales
before few hours of 1:15 pm is high as compared to few hours after 1:15 pm.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/2928880d-1283-4e09-b5cc-531ab098257e)

Now, from the graph we can see that the maximum sale in the entire year is in December and
January. The reason behind the increase in the sales is the number of competitions held
during this period is more. As many of the competitions are held during this period, the
number of sports apparels required for these increases. So, due to this there will be an
increase in the sales of the product. So, a company should be having enough products during
this time so that they can fulfill the customers’ orders. Also, from August to November the
sales remains the same. The reason is the school, universities and colleges reopen, so due to
this there will be an increase in the sales of the product. Also, there is no sale in the month of
April. The simple reason for this is during this time there will be an exam in the schools and
colleges.

![image](https://github.com/Satya-bit/Triumph-Sportswear-Logistic-Regression-Based-Classification/assets/70309925/72d837e7-d499-4851-8456-231bff98dcea)

The following graph shows us the weekly sales data of the company happening on amazon.
This too has been extracted the same way from the order date. On Y axis there are numbers
of sales and on X axis there are number of days of week starting from Sunday and ending on
Saturday i.e., 0 to 6 respectively on X axis respectively. As we can see on weekends the sales
are high. It is near 25000 for given data. This can be because people are having holidays from
their work and they have more time to explore household things including clothing. So, to
increase sales the company should give more offers on weekends. If new products are to be
introduced it should be introduced on weekends for maximum profits. On Wednesday the
sales are minimum. The reason for this is that the company observes a holiday on Tuesday
due to the power staggering happening in the industrial park it is located in.

**=>Conclusions**

From the above observations we can conclude that company has good amount of revenue and
sales. But to improve they need to work on the promotions strategies to retain their buyers.
They need to work on some sort of schemes to make the customer want to come back to their
product. Further they also lack behind in providing proper customer support which needs to
be looked after and managed properly. Data wise they need to collect information on to why
the refunds are coming and back conclude a pattern as to what is happening to their products
once they reach the customers.
