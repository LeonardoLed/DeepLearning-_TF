# DeepLearning-_TF
# DeepReg
The next code presents a new model hybrid to Predict Transcription Factors.

It has the next architecture:
a) A 4CNN Module
b) A BiLSTM Module
C) A Attetion Mechanism Module


![arch](https://github.com/LeonardoLed/DeepLearning-_TF/assets/33387734/3257648e-890f-4a30-b607-47cd893732e3)



The metrics reach out in this model, no present overfitting and we report the next table results:

| Metric     | DeepReg v2 | DeepReg v1      | DeepTFactor  |
| :------------ |   :---:       |:---:       |--------: |
| Accuracy    | 0.9828     | 0.9742         | 0.9773  |
| Precision   | 0.9512      | 0.9923         | 0.9656   |
| Recall      | 0.9638  | 0.977        | 0.9428  |
| Specificity | 0.9876       | 0.9591         | 0.9888   |
| F1-Score    | 0.9574    | 0.9846         | 0.9541   |
| MCC      | 0.9405 | 0.9066         | 0.9392 |

DeepReg v2 is retrained with Trembl and SwissProt.
On Trembl we took all sequences with 4 and 5 score Annotation.
We decided to take this new model to generalize and correct possible predictive errors.

The TFNet is not comparable with this methods because the model metrics was taken for few proteoms examples.


Inference

To make your own predictions you only need as inpur the fasta file , the model and the weights (the .h5 file) and the tokenizer pickle.  <br/>
Change your file fasta path in the main method (in .py script) in order to inference new predictions. 

Training Model

To traning model from Scratch, you can see "ScratchModel" .
You can run the traning process to Deep Reg changes everything you want.






