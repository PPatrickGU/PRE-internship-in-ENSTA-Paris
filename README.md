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




### PLan

The whole work of this intership can be divided into two parts: Data generation & Models creation.

#### Data generation
In the part of Data generation, two soaftwares are used: Blender and Unreal Engine. Blender is a professional, free and open-source 3D computer graphics software used for creating animated films, visual effects, art and video games, etc. Ocean parameters such as choppiness, wave scale, wind velocity, and wave alignment areadded in the python script to  add the complexity of the sea states and to better simulate the sea wave models. For more details, see [[Blender]](https://github.com/PPatrickGU/PRE-internship-in-ENSTA-Paris/tree/master/Data-creation/Blender)

Even though the simulation of waves offered by blender helps a lot, many physical parameters can not be well simulated in blender, which may influence our training results. I have a try with a more powerful and more realistic platform: Unreal engine . Unreal Engine is the world' s most open and advanced real-time 3D creation platform for photo-real visuals and immersive experiences. The physique parameters can be saved during the simulation. For the difficulty of the creation of ocean in UE4, I finally return to Blender for help. For more details, see [[Unreal Engine]](https://github.com/PPatrickGU/PRE-internship-in-ENSTA-Paris/tree/master/Data-creation/Unreal%20Engine)

#### Model creation
16 models based on Nazar's work are created. GRU - Gated recurrent unit networks, Attention mechanism are used. The structure of the extent models are changed from sequence length of size 1 to sequence length of size N in order to make full use of Attention Mechanism. Transformer Model are also considered to be used. An overview of all the models above is given, and the proposed model created in this internship are also analyzed. For more details, see [[Unreal Engine]](https://github.com/PPatrickGU/PRE-internship-in-ENSTA-Paris/tree/master/Modules)

### Software required
In this work, the following softwares are used: Blender 2.83+, Anaconda and pytorch.

#### Blender 2.83
For Blender 2.83+, please check Ubuntu Software to find the correct version of Blender and install it directly. The Python script works on the version 2.83+(2.9. included), if the version of Blender is not correspondant, the script should also be changed. 

#### Anaconda
Anaconda can be download easily from the official website: [[link]](https://www.anaconda.com/products/individual). After the installation, the first thing we need to do is to transplant the environment.
There are two environment files here:

    environment.yml & requirements.txt

To import the environment, we need to go to the directory where the two files are located:
   
    conda env create -f environment.yml
    pip install -r requirements.txt
   
To export the environments:
    
    conda env export > environment.yml
    pip freeze > requirements.txt
    
To activate the environments:

    conda activate <name_of_environment>
    conda activate dajing
