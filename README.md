The main objective of the project is to understand the mechanisms that determine why certain transcription factors (TFs) bind to a specific region of DNA, defined by E-box sequences, rather than others. 
E-box sequences are fundamental in gene regulation. Understanding why some TFs adhere to this particular sequence and not others is crucial, 
as it will shed light on the regulation of specific genes and their impact on biological processes.
To achieve this purpose, deep neural networks will be employed, specifically recurrent neural networks. By allowing an algorithm to directly analyze the sequence, 
it is expected to autonomously discover the determinants of this binding. Instead of categorizing TF binding into binary categories, the project opts for an approach based on "confidence levels", 
which indicate the probability of TF binding. These levels are based on the amount of ChIP-seq experiments, a biological marker, that show TF binding relative to the total number of experiments. 
This choice is especially valuable, as the project has multiple ChIP-seq experiments, both real and synthetically generated from possible DNA sequences. Instead of setting an arbitrary threshold for "positive" 
experiments, this approach allows for a more precise quantification of TF binding using regression neural networks.
