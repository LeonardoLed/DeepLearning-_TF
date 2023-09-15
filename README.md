# DeepLearning-_TF
The next code presents a new model hybrid to Predict Transcription Factors.

It has the next architecture:
a) A 4CNN Module
b) A BiLSTM Module
C) A Attetion Mechanism Module


![arch](https://github.com/LeonardoLed/DeepLearning-_TF/assets/33387734/3257648e-890f-4a30-b607-47cd893732e3)



The metrics reach out in this model, no present overfitting and we report the next table results:

| Metric     | DeepReg      | DeepTFactor  |
| :------------ |   :---:       | --------: |
| Accuracy        | 0.9742         | 0.9773  |
| Precision        | 0.9923         | 0.9656   |
| Recall        | 0.977        | 0.9428  |
| Specificity        | 0.9591         | 0.9888   |
| F1-Score        | 0.9846         | 0.9541   |
| MCC       | 0.9066         | 0.9392 |

To Download Data, you can see the Data Dir, and you find:

a) AN: Aspergillus nidulans, genome and true predictions from DeepReg <br/>
b) NC: Neurospora crassa, genome and true predictions from DeepReg  <br/>
c) SC: Saccharomyces cerevisiae, genome and true predictions from DeepReg  <br/>
d) Experimental.dat: the file that contains all experimental TFs from AN, NC Y SC <br/>
e) ListID_TFs: The ID List from UNIPROT Swissprot Database to create training DB of TF  <br/>

Inference

To make your own predictions you only need as inpur the fasta file , the model and the weights (the .h5 file) and the tokenizer pickle.  <br/>
Change your file fasta path in the main method (in .py script) in order to inference a new predictions. 



