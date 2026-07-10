Title:
Classification of ECG signals with a hybrid approach of scatter transform and deep convolutional net

Abstract:
The goal of automated classification of ECG signals is to enable a fast and objective way to 
diagnose heart related diseases, which will allow saving costs in modern healthcare systems. 
In order to improve upon accuracy, the field is always searching for better ways to extract features from the ECG-signal. 

Starting with handcrafted features that utilize the knowledge of domain experts as well as general purpose statistical properties, recent approaches investigate the automatic discovery of features via deep neural networks. 
 
Cardiovascular arrhythmias, however, manifest themselves in features on different time scales: small scale morphological features, such as missing P-waves, and rhythmical features apparent on heart rate scales. 
 
We tackle this problem by combining information gathered from deep neural networks with features computed by cascades of complex wavelet transforms alternating with a modulus non-linearity, called a scattering transform. The latter has the advantage of being derived from theoretical considerations, which exploit the local structure of ECGs. This approach also enables partial explain-ability, rendering our system a grey box model. 

The motivation for our ansatz is to augment the classifier with a deep net to capture information that is missed by the scattering transform. The most important of which are features that require the information of multiple channels simultaneously, as well as features that are learned specifically for their morphological properties. 

Our joint approach significantly reduces the number of trainable parameters while maintaining high expressiveness, reducing the risks of overfitting and making it a promising candidate for efficient but accurate time series classification. 

For evaluation, we submitted our model in the unofficial phase in the PhysioNet/Computing in Cardiology Challenge 2020. 

Our proof of concept model, achieved a $F_2$-score of 0.617 and a $G_2$-score of 0.404, resulting in a geometric mean of 0.499 on the validation set.
