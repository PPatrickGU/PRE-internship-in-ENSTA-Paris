# PRe internship in ENSTA Paris
***Author:*** [Dajing GU](https://github.com/PPatrickGU)


 Deep learning for predicting ship motion from images, pitches and rolls


### Abstract

With the continuous improvement of computing power, more and more autonomous driving systems involving artificial intelligence have now been developed, which can meet people's requirements for higher efficiency and more comfort. Marine transportation plays an important role in the increasing globalization. I worked on the problem of predicting ship motion from images of the sea surface and previous pitches and rolls.

First of all, a data set is created by using 3D graphics generator Blender, which simulates the ship's  movement through sea waves and offers us the information of the boat (the parameter of pitch and roll in our case). A more professional physical platform: Unreal Engine has also been tried. Data streams were structured in sequences and normalized for further processing. Then 16 different neural networks were developed based on previous work in order to better handle the time-ordered image-data sequences. Gated Recurrent Unit (GRU), Attention mechanism and Transformer Model have been introduced in the existent models of convolutional neural networks (CNN) and long short-term memory networks (LSTM). The models are tested and the best model hyper-parameters have been found by the Hyperband algorithm. All experiment results are analyzed. Also, some possible improvements are suggested.

Keywords:  Deep learning, Ship motion, Time series forecasting, Blender, Unreal Engine, LSTM, GRU, Attention, Transformer 

> Deep learning for predicting ship motion from images, pitches and data.
> Dajing GU, 2020.
> [[Report]](https://drive.google.com/file/d/1f1X34hPGru_1TWm-Wwvk2vFtxVj1hsTX/view?usp=sharing)

Project data & results: [[Google drive]]()

This work is based on the work of Nazar,  some information can be found here: [[link]](https://github.com/Nazotron1923/Deep_learning_models_for_ship_motion_prediction_from_images)
