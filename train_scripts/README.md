# Fishial.ai

This notebook shows insights of training experiments. 

---

![Animation](../imgs/boardProjector.gif)


**Trained Model Report**



1. **Data**

In this experiment, we decomposed the entire exported Fishial dataset into 4 datasets (Train/Test/Remain/Out of Class). Classes with the largest number of images with the ODM flag were selected as target classes for recognition.



    1. **Train/Test**

As a training sample, 75 classes were selected that have the following characteristics: The maximum number of photos is 211, the number of photos for testing is 20% but not less than 15.



    2. **‘Remain’**

	This dataset had all other photos for the selected classes including and not including the ODM flag.



    3. **‘Out of Class’**

	This dataset had all out-of-class images (about 300 different classes) with a low number of instances for each class. This set was chosen in order to check the quality of work on classes that do not participate in the learning process, which thereby gives us an idea of the ability of a neural network to cluster previously unknown classes in hyperspace.

	



2. **Model description.**

Feature Network (backbone): **ResNet18**

Classification Layer (embedding): **128 neurons**

Loss: **Quadruplet loss**

Setup: 

![alt_text](../imgs/image1.png)


3. **Train results **

To check the quality of the model, the **F1-score** method was chosen. Vector neural networks do not return the probability of belonging to a certain class, but only a vector representation of the image, and the next step is to search for the most similar image in the desired class among all elements in the database, thus such networks can be called self-learning, provided that new ones are added to the database validated images, but when new classes are added, retraining is still required. 

The L1 metric (Manhattan distance) was used to search for the most similar image from the database.

For additional improvement (elimination of false positives on classes that did not take part in training), a method was developed to find the best threshold of acceptable distance with the optimal value.

![alt_text](../imgs/image2.png "image_tooltip")



<table>
  <tr>
   <td><strong>Dataset</strong>
   </td>
   <td><strong>Threshold</strong>
   </td>
   <td><strong> F1 score</strong>
   </td>
  </tr>
  <tr>
   <td>Train
   </td>
   <td>36.50303200089144
   </td>
   <td>1.0
   </td>
  </tr>
  <tr>
   <td>Test
   </td>
   <td>43.02331613758524
   </td>
   <td>0.962000962000962
   </td>
  </tr>
  <tr>
   <td>Remain
   </td>
   <td>30.634776277867015
   </td>
   <td>1.0
   </td>
  </tr>
  <tr>
   <td>Out of class
   </td>
   <td>0.0
   </td>
   <td>0.0
   </td>
  </tr>
</table>


Only the training data set was used as a** knowledge base** and distances 0 were cut off because most likely these distances appear in the case of identical photos.

In the table for the "Out of class" data set, we can notice that the metric has a value of 0, all because there cannot be true positive (**TP**) results in this set, which means that the metric is not representative. The number of nearest neighbors was: **5**

**TRAIN**

In order to better evaluate the mechanics of the trained network, we present the following graphs: A histogram of the distribution of distances for the whole data set and a change in the truth map depending on the distance threshold.

**coefficient variance: 0.17094201770111886**

**68–95–99.7 rule: |0.6902770233568712|0.9400325909831613|0.9917436175991309|**



![alt_text](../imgs/image3.png "image_tooltip")


![alt_text](../imgs/image4.png "image_tooltip")


For the training dataset, as we can see, the histogram looks good according to the Gaussian distribution, which means that for this data, the neural network performs very well.

**TEST**

**coefficient variance:** 0.25709614804267034

**68–95–99.7 rule:** |0.7024369388627618|0.904232578024797|0.9747755451047456|


![alt_text](../imgs/image12.png "image_tooltip")


![alt_text](../imgs/image13.png "image_tooltip")



**Remain**

**coefficient variance: 0.2775085208648267**

**68–95–99.7 rule: |0.6155140973488568|0.8435965773600785|0.9694206761116566|**


![alt_text](../imgs/image6.png "image_tooltip")

![alt_text](../imgs/image7.png "image_tooltip")


**Out of class**

In this data set, all elements were not included in the selected classes, which means we can determine with what probability at what threshold level our model will be wrong on out-of-class data.

As we can see the average value is significantly higher than on the previous three datasets **44.874** vs. **23.984, 26.446, 30.408** for training, testing and others respectively which indicates that the network has not overfitting and the resulting features in the vector have a strong dependence.

**coefficient variance: 0.1876613292578638**

**68–95–99.7 rule: |0.6476987447698744|0.9682008368200837|1.0|**



![alt_text](../imgs/image8.png "image_tooltip")


![alt_text](../imgs/image9.png "image_tooltip")


**Confusion matrix**



![alt_text](../imgs/image10.png "image_tooltip")
![alt_text](../imgs/image11.png "image_tooltip")


**Distribution within classes for different datasets**


<table>
  <tr>
   <td>Name
   </td>
   <td colspan="4" ><strong>Train</strong>
   </td>
   <td colspan="4" ><strong>Test</strong>
   </td>
   <td colspan="4" ><strong>Remain</strong>
   </td>
  </tr>
  <tr>
   <td>
   </td>
   <td><strong>CNT</strong>
   </td>
   <td><strong>STD</strong>
   </td>
   <td><strong>MEAN</strong>
   </td>
   <td><strong>MEDIAN</strong>
   </td>
   <td><strong>CNT</strong>
   </td>
   <td><strong>STD</strong>
   </td>
   <td><strong>MEAN</strong>
   </td>
   <td><strong>MEDIAN</strong>
   </td>
   <td><strong>CNT</strong>
   </td>
   <td><strong>STD</strong>
   </td>
   <td><strong>MEAN</strong>
   </td>
   <td><strong>MEDIAN</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Acanthocybium solandri</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.59</p>

   </td>
   <td><p style="text-align: right">
38.51</p>

   </td>
   <td><p style="text-align: right">
38.07</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
10.18</p>

   </td>
   <td><p style="text-align: right">
41.09</p>

   </td>
   <td><p style="text-align: right">
39.89</p>

   </td>
   <td><p style="text-align: right">
19182</p>

   </td>
   <td><p style="text-align: right">
11.35</p>

   </td>
   <td><p style="text-align: right">
48.67</p>

   </td>
   <td><p style="text-align: right">
47.36</p>

   </td>
  </tr>
  <tr>
   <td><strong>Lutjanus synagris</strong>
   </td>
   <td><p style="text-align: right">
21756</p>

   </td>
   <td><p style="text-align: right">
6.16</p>

   </td>
   <td><p style="text-align: right">
33.67</p>

   </td>
   <td><p style="text-align: right">
33.43</p>

   </td>
   <td><p style="text-align: right">
1332</p>

   </td>
   <td><p style="text-align: right">
9.04</p>

   </td>
   <td><p style="text-align: right">
35.19</p>

   </td>
   <td><p style="text-align: right">
34.12</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Salvelinus fontinalis</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.53</p>

   </td>
   <td><p style="text-align: right">
38.61</p>

   </td>
   <td><p style="text-align: right">
38.49</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
7.4</p>

   </td>
   <td><p style="text-align: right">
37.73</p>

   </td>
   <td><p style="text-align: right">
37.12</p>

   </td>
   <td><p style="text-align: right">
32942</p>

   </td>
   <td><p style="text-align: right">
14.27</p>

   </td>
   <td><p style="text-align: right">
52.1</p>

   </td>
   <td><p style="text-align: right">
50.21</p>

   </td>
  </tr>
  <tr>
   <td><strong>Centropristis striata</strong>
   </td>
   <td><p style="text-align: right">
11130</p>

   </td>
   <td><p style="text-align: right">
6.37</p>

   </td>
   <td><p style="text-align: right">
36.2</p>

   </td>
   <td><p style="text-align: right">
35.98</p>

   </td>
   <td><p style="text-align: right">
650</p>

   </td>
   <td><p style="text-align: right">
10.45</p>

   </td>
   <td><p style="text-align: right">
47.07</p>

   </td>
   <td><p style="text-align: right">
47.56</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Centropomus undecimalis</strong>
   </td>
   <td><p style="text-align: right">
9900</p>

   </td>
   <td><p style="text-align: right">
5.84</p>

   </td>
   <td><p style="text-align: right">
33.67</p>

   </td>
   <td><p style="text-align: right">
33.19</p>

   </td>
   <td><p style="text-align: right">
600</p>

   </td>
   <td><p style="text-align: right">
10.31</p>

   </td>
   <td><p style="text-align: right">
38.05</p>

   </td>
   <td><p style="text-align: right">
36.16</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Micropterus punctulatus</strong>
   </td>
   <td><p style="text-align: right">
23870</p>

   </td>
   <td><p style="text-align: right">
6.24</p>

   </td>
   <td><p style="text-align: right">
34.25</p>

   </td>
   <td><p style="text-align: right">
33.88</p>

   </td>
   <td><p style="text-align: right">
1406</p>

   </td>
   <td><p style="text-align: right">
8.22</p>

   </td>
   <td><p style="text-align: right">
39.51</p>

   </td>
   <td><p style="text-align: right">
38.22</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Micropterus dolomieu</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.57</p>

   </td>
   <td><p style="text-align: right">
37.73</p>

   </td>
   <td><p style="text-align: right">
37.54</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
12.57</p>

   </td>
   <td><p style="text-align: right">
44.22</p>

   </td>
   <td><p style="text-align: right">
42.49</p>

   </td>
   <td><p style="text-align: right">
232806</p>

   </td>
   <td><p style="text-align: right">
14.56</p>

   </td>
   <td><p style="text-align: right">
50.69</p>

   </td>
   <td><p style="text-align: right">
47.97</p>

   </td>
  </tr>
  <tr>
   <td><strong>Cyprinus carpio</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.09</p>

   </td>
   <td><p style="text-align: right">
37.96</p>

   </td>
   <td><p style="text-align: right">
37.81</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
13.13</p>

   </td>
   <td><p style="text-align: right">
46.67</p>

   </td>
   <td><p style="text-align: right">
43.74</p>

   </td>
   <td><p style="text-align: right">
17556</p>

   </td>
   <td><p style="text-align: right">
9.93</p>

   </td>
   <td><p style="text-align: right">
43.6</p>

   </td>
   <td><p style="text-align: right">
42.14</p>

   </td>
  </tr>
  <tr>
   <td><strong>Rutilus rutilus</strong>
   </td>
   <td><p style="text-align: right">
21756</p>

   </td>
   <td><p style="text-align: right">
6.47</p>

   </td>
   <td><p style="text-align: right">
36.36</p>

   </td>
   <td><p style="text-align: right">
35.76</p>

   </td>
   <td><p style="text-align: right">
1332</p>

   </td>
   <td><p style="text-align: right">
8.22</p>

   </td>
   <td><p style="text-align: right">
40.9</p>

   </td>
   <td><p style="text-align: right">
40.44</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lutjanus campechanus</strong>
   </td>
   <td><p style="text-align: right">
5256</p>

   </td>
   <td><p style="text-align: right">
6.62</p>

   </td>
   <td><p style="text-align: right">
36.92</p>

   </td>
   <td><p style="text-align: right">
36.74</p>

   </td>
   <td><p style="text-align: right">
342</p>

   </td>
   <td><p style="text-align: right">
8</p>

   </td>
   <td><p style="text-align: right">
40.93</p>

   </td>
   <td><p style="text-align: right">
40.05</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ameiurus nebulosus</strong>
   </td>
   <td><p style="text-align: right">
10302</p>

   </td>
   <td><p style="text-align: right">
5.57</p>

   </td>
   <td><p style="text-align: right">
38.88</p>

   </td>
   <td><p style="text-align: right">
38.75</p>

   </td>
   <td><p style="text-align: right">
600</p>

   </td>
   <td><p style="text-align: right">
13.69</p>

   </td>
   <td><p style="text-align: right">
52.79</p>

   </td>
   <td><p style="text-align: right">
52.35</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Mustelus canis</strong>
   </td>
   <td><p style="text-align: right">
420</p>

   </td>
   <td><p style="text-align: right">
4.65</p>

   </td>
   <td><p style="text-align: right">
34.71</p>

   </td>
   <td><p style="text-align: right">
34.61</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
8.66</p>

   </td>
   <td><p style="text-align: right">
47.43</p>

   </td>
   <td><p style="text-align: right">
46.56</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Oncorhynchus clarkii</strong>
   </td>
   <td><p style="text-align: right">
14520</p>

   </td>
   <td><p style="text-align: right">
5.86</p>

   </td>
   <td><p style="text-align: right">
36.42</p>

   </td>
   <td><p style="text-align: right">
36.23</p>

   </td>
   <td><p style="text-align: right">
870</p>

   </td>
   <td><p style="text-align: right">
6.64</p>

   </td>
   <td><p style="text-align: right">
41.61</p>

   </td>
   <td><p style="text-align: right">
41.04</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Coryphaena hippurus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.28</p>

   </td>
   <td><p style="text-align: right">
37.3</p>

   </td>
   <td><p style="text-align: right">
37.03</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
17.65</p>

   </td>
   <td><p style="text-align: right">
51.13</p>

   </td>
   <td><p style="text-align: right">
47.11</p>

   </td>
   <td><p style="text-align: right">
391250</p>

   </td>
   <td><p style="text-align: right">
11.61</p>

   </td>
   <td><p style="text-align: right">
49.85</p>

   </td>
   <td><p style="text-align: right">
48.63</p>

   </td>
  </tr>
  <tr>
   <td><strong>Lepomis cyanellus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.25</p>

   </td>
   <td><p style="text-align: right">
33.38</p>

   </td>
   <td><p style="text-align: right">
32.96</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.58</p>

   </td>
   <td><p style="text-align: right">
41.49</p>

   </td>
   <td><p style="text-align: right">
41.53</p>

   </td>
   <td><p style="text-align: right">
342</p>

   </td>
   <td><p style="text-align: right">
29.96</p>

   </td>
   <td><p style="text-align: right">
53.05</p>

   </td>
   <td><p style="text-align: right">
39.12</p>

   </td>
  </tr>
  <tr>
   <td><strong>Scomber scombrus</strong>
   </td>
   <td><p style="text-align: right">
4970</p>

   </td>
   <td><p style="text-align: right">
5.98</p>

   </td>
   <td><p style="text-align: right">
35.75</p>

   </td>
   <td><p style="text-align: right">
35.65</p>

   </td>
   <td><p style="text-align: right">
306</p>

   </td>
   <td><p style="text-align: right">
8.52</p>

   </td>
   <td><p style="text-align: right">
40.93</p>

   </td>
   <td><p style="text-align: right">
41.05</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ctenopharyngodon idella</strong>
   </td>
   <td><p style="text-align: right">
4692</p>

   </td>
   <td><p style="text-align: right">
5.1</p>

   </td>
   <td><p style="text-align: right">
35.18</p>

   </td>
   <td><p style="text-align: right">
35.02</p>

   </td>
   <td><p style="text-align: right">
272</p>

   </td>
   <td><p style="text-align: right">
7.46</p>

   </td>
   <td><p style="text-align: right">
36.68</p>

   </td>
   <td><p style="text-align: right">
35.8</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Cichla ocellaris</strong>
   </td>
   <td><p style="text-align: right">
14762</p>

   </td>
   <td><p style="text-align: right">
6.09</p>

   </td>
   <td><p style="text-align: right">
34.35</p>

   </td>
   <td><p style="text-align: right">
33.98</p>

   </td>
   <td><p style="text-align: right">
870</p>

   </td>
   <td><p style="text-align: right">
8.84</p>

   </td>
   <td><p style="text-align: right">
39.64</p>

   </td>
   <td><p style="text-align: right">
38.96</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Belone belone</strong>
   </td>
   <td><p style="text-align: right">
870</p>

   </td>
   <td><p style="text-align: right">
5.5</p>

   </td>
   <td><p style="text-align: right">
35.73</p>

   </td>
   <td><p style="text-align: right">
35.42</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
11.96</p>

   </td>
   <td><p style="text-align: right">
49.08</p>

   </td>
   <td><p style="text-align: right">
45.68</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepomis microlophus</strong>
   </td>
   <td><p style="text-align: right">
14762</p>

   </td>
   <td><p style="text-align: right">
5.74</p>

   </td>
   <td><p style="text-align: right">
34.73</p>

   </td>
   <td><p style="text-align: right">
34.55</p>

   </td>
   <td><p style="text-align: right">
870</p>

   </td>
   <td><p style="text-align: right">
8.92</p>

   </td>
   <td><p style="text-align: right">
37.26</p>

   </td>
   <td><p style="text-align: right">
35.82</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Archosargus probatocephalus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.68</p>

   </td>
   <td><p style="text-align: right">
36.43</p>

   </td>
   <td><p style="text-align: right">
36.04</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.38</p>

   </td>
   <td><p style="text-align: right">
38.18</p>

   </td>
   <td><p style="text-align: right">
37.21</p>

   </td>
   <td><p style="text-align: right">
15500</p>

   </td>
   <td><p style="text-align: right">
16.04</p>

   </td>
   <td><p style="text-align: right">
50.01</p>

   </td>
   <td><p style="text-align: right">
46.34</p>

   </td>
  </tr>
  <tr>
   <td><strong>Cyprinus carpio carpio</strong>
   </td>
   <td><p style="text-align: right">
1190</p>

   </td>
   <td><p style="text-align: right">
4.61</p>

   </td>
   <td><p style="text-align: right">
35.42</p>

   </td>
   <td><p style="text-align: right">
35.42</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
9.66</p>

   </td>
   <td><p style="text-align: right">
50.09</p>

   </td>
   <td><p style="text-align: right">
48.82</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Pomoxis annularis</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
5.99</p>

   </td>
   <td><p style="text-align: right">
35.19</p>

   </td>
   <td><p style="text-align: right">
34.98</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
7.05</p>

   </td>
   <td><p style="text-align: right">
40.52</p>

   </td>
   <td><p style="text-align: right">
40.44</p>

   </td>
   <td><p style="text-align: right">
12</p>

   </td>
   <td><p style="text-align: right">
14.51</p>

   </td>
   <td><p style="text-align: right">
59.31</p>

   </td>
   <td><p style="text-align: right">
63.1</p>

   </td>
  </tr>
  <tr>
   <td><strong>Sander vitreus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.91</p>

   </td>
   <td><p style="text-align: right">
38.86</p>

   </td>
   <td><p style="text-align: right">
38.79</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.12</p>

   </td>
   <td><p style="text-align: right">
43.33</p>

   </td>
   <td><p style="text-align: right">
42.63</p>

   </td>
   <td><p style="text-align: right">
6</p>

   </td>
   <td><p style="text-align: right">
1.06</p>

   </td>
   <td><p style="text-align: right">
41.56</p>

   </td>
   <td><p style="text-align: right">
41.96</p>

   </td>
  </tr>
  <tr>
   <td><strong>Perca fluviatilis</strong>
   </td>
   <td><p style="text-align: right">
20592</p>

   </td>
   <td><p style="text-align: right">
5.86</p>

   </td>
   <td><p style="text-align: right">
35.99</p>

   </td>
   <td><p style="text-align: right">
35.75</p>

   </td>
   <td><p style="text-align: right">
1190</p>

   </td>
   <td><p style="text-align: right">
8.22</p>

   </td>
   <td><p style="text-align: right">
41.4</p>

   </td>
   <td><p style="text-align: right">
40.62</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Esox masquinongy</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
7.13</p>

   </td>
   <td><p style="text-align: right">
34.8</p>

   </td>
   <td><p style="text-align: right">
34.25</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
13.35</p>

   </td>
   <td><p style="text-align: right">
42.02</p>

   </td>
   <td><p style="text-align: right">
39.86</p>

   </td>
   <td><p style="text-align: right">
6972</p>

   </td>
   <td><p style="text-align: right">
14.49</p>

   </td>
   <td><p style="text-align: right">
58.45</p>

   </td>
   <td><p style="text-align: right">
57.28</p>

   </td>
  </tr>
  <tr>
   <td><strong>Paralichthys lethostigma</strong>
   </td>
   <td><p style="text-align: right">
11556</p>

   </td>
   <td><p style="text-align: right">
6.43</p>

   </td>
   <td><p style="text-align: right">
35.63</p>

   </td>
   <td><p style="text-align: right">
35.18</p>

   </td>
   <td><p style="text-align: right">
702</p>

   </td>
   <td><p style="text-align: right">
8.67</p>

   </td>
   <td><p style="text-align: right">
38.93</p>

   </td>
   <td><p style="text-align: right">
38.51</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepisosteus osseus</strong>
   </td>
   <td><p style="text-align: right">
9120</p>

   </td>
   <td><p style="text-align: right">
6.39</p>

   </td>
   <td><p style="text-align: right">
36.48</p>

   </td>
   <td><p style="text-align: right">
36.32</p>

   </td>
   <td><p style="text-align: right">
506</p>

   </td>
   <td><p style="text-align: right">
7.31</p>

   </td>
   <td><p style="text-align: right">
44.44</p>

   </td>
   <td><p style="text-align: right">
44.52</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Esox niger</strong>
   </td>
   <td><p style="text-align: right">
19460</p>

   </td>
   <td><p style="text-align: right">
7.14</p>

   </td>
   <td><p style="text-align: right">
37.27</p>

   </td>
   <td><p style="text-align: right">
37.52</p>

   </td>
   <td><p style="text-align: right">
1190</p>

   </td>
   <td><p style="text-align: right">
7.9</p>

   </td>
   <td><p style="text-align: right">
41.24</p>

   </td>
   <td><p style="text-align: right">
41.25</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Oncorhynchus tshawytscha</strong>
   </td>
   <td><p style="text-align: right">
12432</p>

   </td>
   <td><p style="text-align: right">
5.96</p>

   </td>
   <td><p style="text-align: right">
38.26</p>

   </td>
   <td><p style="text-align: right">
37.86</p>

   </td>
   <td><p style="text-align: right">
702</p>

   </td>
   <td><p style="text-align: right">
8.5</p>

   </td>
   <td><p style="text-align: right">
50.98</p>

   </td>
   <td><p style="text-align: right">
50.68</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Pomatomus saltatrix</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.47</p>

   </td>
   <td><p style="text-align: right">
37.64</p>

   </td>
   <td><p style="text-align: right">
37.35</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.38</p>

   </td>
   <td><p style="text-align: right">
48</p>

   </td>
   <td><p style="text-align: right">
47.93</p>

   </td>
   <td><p style="text-align: right">
510510</p>

   </td>
   <td><p style="text-align: right">
12.09</p>

   </td>
   <td><p style="text-align: right">
54.22</p>

   </td>
   <td><p style="text-align: right">
53.22</p>

   </td>
  </tr>
  <tr>
   <td><strong>Pomoxis nigromaculatus</strong>
   </td>
   <td><p style="text-align: right">
20592</p>

   </td>
   <td><p style="text-align: right">
6.2</p>

   </td>
   <td><p style="text-align: right">
35.86</p>

   </td>
   <td><p style="text-align: right">
35.49</p>

   </td>
   <td><p style="text-align: right">
1190</p>

   </td>
   <td><p style="text-align: right">
8.88</p>

   </td>
   <td><p style="text-align: right">
37.6</p>

   </td>
   <td><p style="text-align: right">
36.5</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Scomberomorus maculatus</strong>
   </td>
   <td><p style="text-align: right">
26732</p>

   </td>
   <td><p style="text-align: right">
7.45</p>

   </td>
   <td><p style="text-align: right">
35.92</p>

   </td>
   <td><p style="text-align: right">
35.89</p>

   </td>
   <td><p style="text-align: right">
1640</p>

   </td>
   <td><p style="text-align: right">
12.76</p>

   </td>
   <td><p style="text-align: right">
38.59</p>

   </td>
   <td><p style="text-align: right">
35.64</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Platycephalus fuscus</strong>
   </td>
   <td><p style="text-align: right">
8190</p>

   </td>
   <td><p style="text-align: right">
6.12</p>

   </td>
   <td><p style="text-align: right">
37.77</p>

   </td>
   <td><p style="text-align: right">
37.65</p>

   </td>
   <td><p style="text-align: right">
506</p>

   </td>
   <td><p style="text-align: right">
6.93</p>

   </td>
   <td><p style="text-align: right">
41.65</p>

   </td>
   <td><p style="text-align: right">
41.74</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Perca flavescens</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.47</p>

   </td>
   <td><p style="text-align: right">
37.2</p>

   </td>
   <td><p style="text-align: right">
36.84</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
8.53</p>

   </td>
   <td><p style="text-align: right">
41.66</p>

   </td>
   <td><p style="text-align: right">
40.59</p>

   </td>
   <td><p style="text-align: right">
123552</p>

   </td>
   <td><p style="text-align: right">
9.78</p>

   </td>
   <td><p style="text-align: right">
44.45</p>

   </td>
   <td><p style="text-align: right">
43.1</p>

   </td>
  </tr>
  <tr>
   <td><strong>Seriola dumerili</strong>
   </td>
   <td><p style="text-align: right">
1122</p>

   </td>
   <td><p style="text-align: right">
5.97</p>

   </td>
   <td><p style="text-align: right">
35.65</p>

   </td>
   <td><p style="text-align: right">
35.34</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
14.07</p>

   </td>
   <td><p style="text-align: right">
52.32</p>

   </td>
   <td><p style="text-align: right">
50.07</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Caranx hippos</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.71</p>

   </td>
   <td><p style="text-align: right">
37.13</p>

   </td>
   <td><p style="text-align: right">
36.96</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
15.25</p>

   </td>
   <td><p style="text-align: right">
48.16</p>

   </td>
   <td><p style="text-align: right">
45.64</p>

   </td>
   <td><p style="text-align: right">
172640</p>

   </td>
   <td><p style="text-align: right">
14.53</p>

   </td>
   <td><p style="text-align: right">
52.18</p>

   </td>
   <td><p style="text-align: right">
49.93</p>

   </td>
  </tr>
  <tr>
   <td><strong>Ambloplites rupestris</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6</p>

   </td>
   <td><p style="text-align: right">
34.12</p>

   </td>
   <td><p style="text-align: right">
33.79</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
6.82</p>

   </td>
   <td><p style="text-align: right">
37.38</p>

   </td>
   <td><p style="text-align: right">
36.92</p>

   </td>
   <td><p style="text-align: right">
156</p>

   </td>
   <td><p style="text-align: right">
14.23</p>

   </td>
   <td><p style="text-align: right">
43.25</p>

   </td>
   <td><p style="text-align: right">
41.39</p>

   </td>
  </tr>
  <tr>
   <td><strong>Oncorhynchus mykiss</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.3</p>

   </td>
   <td><p style="text-align: right">
38.26</p>

   </td>
   <td><p style="text-align: right">
37.97</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.89</p>

   </td>
   <td><p style="text-align: right">
46.47</p>

   </td>
   <td><p style="text-align: right">
45.16</p>

   </td>
   <td><p style="text-align: right">
281430</p>

   </td>
   <td><p style="text-align: right">
12.08</p>

   </td>
   <td><p style="text-align: right">
51.03</p>

   </td>
   <td><p style="text-align: right">
49.85</p>

   </td>
  </tr>
  <tr>
   <td><strong>Dicentrarchus labrax</strong>
   </td>
   <td><p style="text-align: right">
15252</p>

   </td>
   <td><p style="text-align: right">
6.12</p>

   </td>
   <td><p style="text-align: right">
37.62</p>

   </td>
   <td><p style="text-align: right">
37.78</p>

   </td>
   <td><p style="text-align: right">
930</p>

   </td>
   <td><p style="text-align: right">
11.17</p>

   </td>
   <td><p style="text-align: right">
47.28</p>

   </td>
   <td><p style="text-align: right">
45.83</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Amia calva</strong>
   </td>
   <td><p style="text-align: right">
26732</p>

   </td>
   <td><p style="text-align: right">
5.8</p>

   </td>
   <td><p style="text-align: right">
39.91</p>

   </td>
   <td><p style="text-align: right">
39.83</p>

   </td>
   <td><p style="text-align: right">
1560</p>

   </td>
   <td><p style="text-align: right">
10.73</p>

   </td>
   <td><p style="text-align: right">
49.32</p>

   </td>
   <td><p style="text-align: right">
48.33</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ameiurus catus</strong>
   </td>
   <td><p style="text-align: right">
702</p>

   </td>
   <td><p style="text-align: right">
4.64</p>

   </td>
   <td><p style="text-align: right">
36.24</p>

   </td>
   <td><p style="text-align: right">
36.16</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
9.27</p>

   </td>
   <td><p style="text-align: right">
49.08</p>

   </td>
   <td><p style="text-align: right">
50.91</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Mycteroperca bonaci</strong>
   </td>
   <td><p style="text-align: right">
6642</p>

   </td>
   <td><p style="text-align: right">
6.57</p>

   </td>
   <td><p style="text-align: right">
33.59</p>

   </td>
   <td><p style="text-align: right">
33.02</p>

   </td>
   <td><p style="text-align: right">
380</p>

   </td>
   <td><p style="text-align: right">
11.06</p>

   </td>
   <td><p style="text-align: right">
43.85</p>

   </td>
   <td><p style="text-align: right">
41.94</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Sphyraena barracuda</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.33</p>

   </td>
   <td><p style="text-align: right">
38.14</p>

   </td>
   <td><p style="text-align: right">
38.1</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.04</p>

   </td>
   <td><p style="text-align: right">
41.87</p>

   </td>
   <td><p style="text-align: right">
41.58</p>

   </td>
   <td><p style="text-align: right">
1167480</p>

   </td>
   <td><p style="text-align: right">
12.72</p>

   </td>
   <td><p style="text-align: right">
52.23</p>

   </td>
   <td><p style="text-align: right">
51.81</p>

   </td>
  </tr>
  <tr>
   <td><strong>Micropterus salmoides</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.29</p>

   </td>
   <td><p style="text-align: right">
36.96</p>

   </td>
   <td><p style="text-align: right">
36.54</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.82</p>

   </td>
   <td><p style="text-align: right">
44.58</p>

   </td>
   <td><p style="text-align: right">
43.35</p>

   </td>
   <td><p style="text-align: right">
382542</p>

   </td>
   <td><p style="text-align: right">
12.23</p>

   </td>
   <td><p style="text-align: right">
49.15</p>

   </td>
   <td><p style="text-align: right">
47.52</p>

   </td>
  </tr>
  <tr>
   <td><strong>Lutjanus griseus</strong>
   </td>
   <td><p style="text-align: right">
12432</p>

   </td>
   <td><p style="text-align: right">
6.4</p>

   </td>
   <td><p style="text-align: right">
37.98</p>

   </td>
   <td><p style="text-align: right">
37.8</p>

   </td>
   <td><p style="text-align: right">
756</p>

   </td>
   <td><p style="text-align: right">
12.2</p>

   </td>
   <td><p style="text-align: right">
53.6</p>

   </td>
   <td><p style="text-align: right">
52.62</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepomis gulosus</strong>
   </td>
   <td><p style="text-align: right">
14762</p>

   </td>
   <td><p style="text-align: right">
5.9</p>

   </td>
   <td><p style="text-align: right">
37.28</p>

   </td>
   <td><p style="text-align: right">
37.18</p>

   </td>
   <td><p style="text-align: right">
930</p>

   </td>
   <td><p style="text-align: right">
9.8</p>

   </td>
   <td><p style="text-align: right">
51.31</p>

   </td>
   <td><p style="text-align: right">
51.01</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Salmo trutta</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.57</p>

   </td>
   <td><p style="text-align: right">
35.46</p>

   </td>
   <td><p style="text-align: right">
34.87</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.06</p>

   </td>
   <td><p style="text-align: right">
39.1</p>

   </td>
   <td><p style="text-align: right">
38.02</p>

   </td>
   <td><p style="text-align: right">
812</p>

   </td>
   <td><p style="text-align: right">
10.98</p>

   </td>
   <td><p style="text-align: right">
42.82</p>

   </td>
   <td><p style="text-align: right">
40.71</p>

   </td>
  </tr>
  <tr>
   <td><strong>Rhincodon typus</strong>
   </td>
   <td><p style="text-align: right">
1406</p>

   </td>
   <td><p style="text-align: right">
6.23</p>

   </td>
   <td><p style="text-align: right">
38.97</p>

   </td>
   <td><p style="text-align: right">
38.69</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
11.18</p>

   </td>
   <td><p style="text-align: right">
46.84</p>

   </td>
   <td><p style="text-align: right">
46.92</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ictalurus punctatus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.21</p>

   </td>
   <td><p style="text-align: right">
38.88</p>

   </td>
   <td><p style="text-align: right">
38.69</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
10.61</p>

   </td>
   <td><p style="text-align: right">
44.08</p>

   </td>
   <td><p style="text-align: right">
42.51</p>

   </td>
   <td><p style="text-align: right">
18090</p>

   </td>
   <td><p style="text-align: right">
11.43</p>

   </td>
   <td><p style="text-align: right">
55.51</p>

   </td>
   <td><p style="text-align: right">
55.42</p>

   </td>
  </tr>
  <tr>
   <td><strong>Morone saxatilis</strong>
   </td>
   <td><p style="text-align: right">
25760</p>

   </td>
   <td><p style="text-align: right">
7.29</p>

   </td>
   <td><p style="text-align: right">
33.41</p>

   </td>
   <td><p style="text-align: right">
32.4</p>

   </td>
   <td><p style="text-align: right">
1560</p>

   </td>
   <td><p style="text-align: right">
12.09</p>

   </td>
   <td><p style="text-align: right">
41.04</p>

   </td>
   <td><p style="text-align: right">
39.92</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Sciaenops ocellatus</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.92</p>

   </td>
   <td><p style="text-align: right">
38.17</p>

   </td>
   <td><p style="text-align: right">
37.73</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
8.89</p>

   </td>
   <td><p style="text-align: right">
43.33</p>

   </td>
   <td><p style="text-align: right">
42.07</p>

   </td>
   <td><p style="text-align: right">
428370</p>

   </td>
   <td><p style="text-align: right">
12.67</p>

   </td>
   <td><p style="text-align: right">
52.36</p>

   </td>
   <td><p style="text-align: right">
50.95</p>

   </td>
  </tr>
  <tr>
   <td><strong>Lutjanus analis</strong>
   </td>
   <td><p style="text-align: right">
4830</p>

   </td>
   <td><p style="text-align: right">
5.23</p>

   </td>
   <td><p style="text-align: right">
34.97</p>

   </td>
   <td><p style="text-align: right">
34.67</p>

   </td>
   <td><p style="text-align: right">
272</p>

   </td>
   <td><p style="text-align: right">
10.84</p>

   </td>
   <td><p style="text-align: right">
50.92</p>

   </td>
   <td><p style="text-align: right">
50.35</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Cynoscion nebulosus</strong>
   </td>
   <td><p style="text-align: right">
23562</p>

   </td>
   <td><p style="text-align: right">
6.76</p>

   </td>
   <td><p style="text-align: right">
39.09</p>

   </td>
   <td><p style="text-align: right">
38.82</p>

   </td>
   <td><p style="text-align: right">
1406</p>

   </td>
   <td><p style="text-align: right">
11.34</p>

   </td>
   <td><p style="text-align: right">
44.27</p>

   </td>
   <td><p style="text-align: right">
42.07</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Morone chrysops</strong>
   </td>
   <td><p style="text-align: right">
18632</p>

   </td>
   <td><p style="text-align: right">
6.16</p>

   </td>
   <td><p style="text-align: right">
34.81</p>

   </td>
   <td><p style="text-align: right">
34.38</p>

   </td>
   <td><p style="text-align: right">
1122</p>

   </td>
   <td><p style="text-align: right">
10.37</p>

   </td>
   <td><p style="text-align: right">
37.22</p>

   </td>
   <td><p style="text-align: right">
35.39</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Thunnus albacares</strong>
   </td>
   <td><p style="text-align: right">
4830</p>

   </td>
   <td><p style="text-align: right">
6.21</p>

   </td>
   <td><p style="text-align: right">
36.7</p>

   </td>
   <td><p style="text-align: right">
36.43</p>

   </td>
   <td><p style="text-align: right">
306</p>

   </td>
   <td><p style="text-align: right">
8.33</p>

   </td>
   <td><p style="text-align: right">
45.44</p>

   </td>
   <td><p style="text-align: right">
45.26</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepomis gibbosus</strong>
   </td>
   <td><p style="text-align: right">
20592</p>

   </td>
   <td><p style="text-align: right">
6.31</p>

   </td>
   <td><p style="text-align: right">
35.44</p>

   </td>
   <td><p style="text-align: right">
35.07</p>

   </td>
   <td><p style="text-align: right">
1190</p>

   </td>
   <td><p style="text-align: right">
9.66</p>

   </td>
   <td><p style="text-align: right">
41.54</p>

   </td>
   <td><p style="text-align: right">
40.91</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Pylodictis olivaris</strong>
   </td>
   <td><p style="text-align: right">
10302</p>

   </td>
   <td><p style="text-align: right">
5.67</p>

   </td>
   <td><p style="text-align: right">
38.29</p>

   </td>
   <td><p style="text-align: right">
38.09</p>

   </td>
   <td><p style="text-align: right">
600</p>

   </td>
   <td><p style="text-align: right">
10.01</p>

   </td>
   <td><p style="text-align: right">
46.14</p>

   </td>
   <td><p style="text-align: right">
44.72</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Morone americana</strong>
   </td>
   <td><p style="text-align: right">
14762</p>

   </td>
   <td><p style="text-align: right">
6.06</p>

   </td>
   <td><p style="text-align: right">
37.34</p>

   </td>
   <td><p style="text-align: right">
37</p>

   </td>
   <td><p style="text-align: right">
870</p>

   </td>
   <td><p style="text-align: right">
9.42</p>

   </td>
   <td><p style="text-align: right">
46.48</p>

   </td>
   <td><p style="text-align: right">
45.27</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Carcharhinus limbatus</strong>
   </td>
   <td><p style="text-align: right">
2162</p>

   </td>
   <td><p style="text-align: right">
5.19</p>

   </td>
   <td><p style="text-align: right">
35.33</p>

   </td>
   <td><p style="text-align: right">
35.38</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
8.57</p>

   </td>
   <td><p style="text-align: right">
40.62</p>

   </td>
   <td><p style="text-align: right">
40.06</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Pogonias cromis</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.17</p>

   </td>
   <td><p style="text-align: right">
37.31</p>

   </td>
   <td><p style="text-align: right">
37.26</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
11.67</p>

   </td>
   <td><p style="text-align: right">
46.54</p>

   </td>
   <td><p style="text-align: right">
45.26</p>

   </td>
   <td><p style="text-align: right">
292140</p>

   </td>
   <td><p style="text-align: right">
14.13</p>

   </td>
   <td><p style="text-align: right">
57.9</p>

   </td>
   <td><p style="text-align: right">
56.86</p>

   </td>
  </tr>
  <tr>
   <td><strong>Abramis brama</strong>
   </td>
   <td><p style="text-align: right">
9312</p>

   </td>
   <td><p style="text-align: right">
6.36</p>

   </td>
   <td><p style="text-align: right">
37.1</p>

   </td>
   <td><p style="text-align: right">
36.7</p>

   </td>
   <td><p style="text-align: right">
552</p>

   </td>
   <td><p style="text-align: right">
7.49</p>

   </td>
   <td><p style="text-align: right">
45.24</p>

   </td>
   <td><p style="text-align: right">
45.26</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepomis macrochirus</strong>
   </td>
   <td><p style="text-align: right">
20592</p>

   </td>
   <td><p style="text-align: right">
5.74</p>

   </td>
   <td><p style="text-align: right">
36.4</p>

   </td>
   <td><p style="text-align: right">
36.1</p>

   </td>
   <td><p style="text-align: right">
1260</p>

   </td>
   <td><p style="text-align: right">
13.82</p>

   </td>
   <td><p style="text-align: right">
48.2</p>

   </td>
   <td><p style="text-align: right">
45.85</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Lepomis auritus</strong>
   </td>
   <td><p style="text-align: right">
22952</p>

   </td>
   <td><p style="text-align: right">
6.79</p>

   </td>
   <td><p style="text-align: right">
35.58</p>

   </td>
   <td><p style="text-align: right">
35.15</p>

   </td>
   <td><p style="text-align: right">
1332</p>

   </td>
   <td><p style="text-align: right">
10.22</p>

   </td>
   <td><p style="text-align: right">
42.7</p>

   </td>
   <td><p style="text-align: right">
41.01</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Balistes capriscus</strong>
   </td>
   <td><p style="text-align: right">
272</p>

   </td>
   <td><p style="text-align: right">
6.43</p>

   </td>
   <td><p style="text-align: right">
36.09</p>

   </td>
   <td><p style="text-align: right">
35.44</p>

   </td>
   <td><p style="text-align: right">
210</p>

   </td>
   <td><p style="text-align: right">
10.61</p>

   </td>
   <td><p style="text-align: right">
51.24</p>

   </td>
   <td><p style="text-align: right">
51.35</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Salvelinus namaycush</strong>
   </td>
   <td><p style="text-align: right">
8372</p>

   </td>
   <td><p style="text-align: right">
5.81</p>

   </td>
   <td><p style="text-align: right">
35.75</p>

   </td>
   <td><p style="text-align: right">
35.51</p>

   </td>
   <td><p style="text-align: right">
506</p>

   </td>
   <td><p style="text-align: right">
12.41</p>

   </td>
   <td><p style="text-align: right">
53.65</p>

   </td>
   <td><p style="text-align: right">
53.06</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Esox lucius</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
7.39</p>

   </td>
   <td><p style="text-align: right">
35.91</p>

   </td>
   <td><p style="text-align: right">
35.4</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.97</p>

   </td>
   <td><p style="text-align: right">
43.58</p>

   </td>
   <td><p style="text-align: right">
42.4</p>

   </td>
   <td><p style="text-align: right">
23562</p>

   </td>
   <td><p style="text-align: right">
13.2</p>

   </td>
   <td><p style="text-align: right">
47.58</p>

   </td>
   <td><p style="text-align: right">
45.27</p>

   </td>
  </tr>
  <tr>
   <td><strong>Epinephelus morio</strong>
   </td>
   <td><p style="text-align: right">
12210</p>

   </td>
   <td><p style="text-align: right">
6.66</p>

   </td>
   <td><p style="text-align: right">
40.08</p>

   </td>
   <td><p style="text-align: right">
39.91</p>

   </td>
   <td><p style="text-align: right">
702</p>

   </td>
   <td><p style="text-align: right">
9.1</p>

   </td>
   <td><p style="text-align: right">
46.82</p>

   </td>
   <td><p style="text-align: right">
45.95</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ameiurus melas</strong>
   </td>
   <td><p style="text-align: right">
8742</p>

   </td>
   <td><p style="text-align: right">
6.04</p>

   </td>
   <td><p style="text-align: right">
35.98</p>

   </td>
   <td><p style="text-align: right">
35.62</p>

   </td>
   <td><p style="text-align: right">
506</p>

   </td>
   <td><p style="text-align: right">
12.16</p>

   </td>
   <td><p style="text-align: right">
45.01</p>

   </td>
   <td><p style="text-align: right">
42.28</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Micropogonias undulatus</strong>
   </td>
   <td><p style="text-align: right">
12432</p>

   </td>
   <td><p style="text-align: right">
5.58</p>

   </td>
   <td><p style="text-align: right">
36.34</p>

   </td>
   <td><p style="text-align: right">
36.21</p>

   </td>
   <td><p style="text-align: right">
756</p>

   </td>
   <td><p style="text-align: right">
12.61</p>

   </td>
   <td><p style="text-align: right">
47.51</p>

   </td>
   <td><p style="text-align: right">
45.86</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Scomberomorus cavalla</strong>
   </td>
   <td><p style="text-align: right">
28392</p>

   </td>
   <td><p style="text-align: right">
6.25</p>

   </td>
   <td><p style="text-align: right">
38.38</p>

   </td>
   <td><p style="text-align: right">
38.09</p>

   </td>
   <td><p style="text-align: right">
1722</p>

   </td>
   <td><p style="text-align: right">
9.85</p>

   </td>
   <td><p style="text-align: right">
44.97</p>

   </td>
   <td><p style="text-align: right">
44.64</p>

   </td>
   <td><p style="text-align: right">
2450</p>

   </td>
   <td><p style="text-align: right">
12.53</p>

   </td>
   <td><p style="text-align: right">
49.85</p>

   </td>
   <td><p style="text-align: right">
47.74</p>

   </td>
  </tr>
  <tr>
   <td><strong>Tilapia sparrmanii</strong>
   </td>
   <td><p style="text-align: right">
7310</p>

   </td>
   <td><p style="text-align: right">
6.21</p>

   </td>
   <td><p style="text-align: right">
32.43</p>

   </td>
   <td><p style="text-align: right">
32.16</p>

   </td>
   <td><p style="text-align: right">
420</p>

   </td>
   <td><p style="text-align: right">
7.83</p>

   </td>
   <td><p style="text-align: right">
30.31</p>

   </td>
   <td><p style="text-align: right">
29.11</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Aplodinotus grunniens</strong>
   </td>
   <td><p style="text-align: right">
9312</p>

   </td>
   <td><p style="text-align: right">
5.77</p>

   </td>
   <td><p style="text-align: right">
36.84</p>

   </td>
   <td><p style="text-align: right">
36.68</p>

   </td>
   <td><p style="text-align: right">
552</p>

   </td>
   <td><p style="text-align: right">
9.49</p>

   </td>
   <td><p style="text-align: right">
44.1</p>

   </td>
   <td><p style="text-align: right">
44.12</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Ictalurus furcatus</strong>
   </td>
   <td><p style="text-align: right">
8742</p>

   </td>
   <td><p style="text-align: right">
6.05</p>

   </td>
   <td><p style="text-align: right">
37.65</p>

   </td>
   <td><p style="text-align: right">
37.34</p>

   </td>
   <td><p style="text-align: right">
506</p>

   </td>
   <td><p style="text-align: right">
16.98</p>

   </td>
   <td><p style="text-align: right">
50.71</p>

   </td>
   <td><p style="text-align: right">
47.61</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td><strong>Paralichthys dentatus</strong>
   </td>
   <td><p style="text-align: right">
8010</p>

   </td>
   <td><p style="text-align: right">
6.3</p>

   </td>
   <td><p style="text-align: right">
35.22</p>

   </td>
   <td><p style="text-align: right">
34.58</p>

   </td>
   <td><p style="text-align: right">
462</p>

   </td>
   <td><p style="text-align: right">
8.91</p>

   </td>
   <td><p style="text-align: right">
40.85</p>

   </td>
   <td><p style="text-align: right">
39.9</p>

   </td>
   <td><p style="text-align: right">
0</p>

   </td>
   <td>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
</table>



<p float="left">
  <img src="https://fishial.ai/static/fishial_logo-2c651a547f55002df228d91f57178377.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2020/08/68e6fe03-e654-4d15-9161-98715ff1f393.png" height="40" /> 
  <img src="https://wp.fishial.ai/wp-content/uploads/2021/01/WYE-Foundation-Full-Color.png" height="40" />
  <img src="https://wp.fishial.ai/wp-content/uploads/2019/08/dotcom-standard.png" height="40" />
</p>


## License

[MIT](https://choosealicense.com/licenses/mit/)

